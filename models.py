import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import accuracy_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract features from BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [cls] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained = True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False


        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Change to (batch_size, channels, frames, height, width)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size = 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size = 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))  # Remove the last dimension


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) 

        # Classification layer
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7) # Assuming 7 emotion classes(sadness, anger...)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # Assuming 3 sentiment classes (negative, neutral, positive)
        ) 


    def forward(self, text_inputs, video_frames, audio_features):
        # Encode text, video, and audio
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )

        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features
        combined_features = torch.cat((text_features, video_features, audio_features), dim=1) # [batch_size, 128 * 3]

        fused_features = self.fusion_layer(combined_features)  # [batch_size, 256]
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotion_output': emotion_output,
            'sentiment_output': sentiment_output
        }

class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Log dataset sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader)}")

        timesstamp = datetime.now().strftime('%b%d_%H-%M-%S')
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timesstamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0


        # very high:1, high: 0.1-0.01, medium: 1e-1, low: 1e-4, vry low: 1e-5
        self.optimizer = torch.optim.Adam([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6}, 
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-4},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay = 1e-5)


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2,     
        )

        self.current_train_losses = None

        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05
        )

    def log_metrics(self, losses, metrics = None, phase = 'train'):
        if phase == 'train':
            self.current_train_losses = losses
        else:
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar(
                'loss/total/val', losses['total'], self.global_step)
            self.writer.add_scalar(
            'loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar(
            'loss/emotion/val', losses['emotion'], self.global_step)
            self.writer.add_scalar(
            'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar(
            'loss/sentiment/val', losses['sentiment'], self.global_step)
        
        if metrics:
            self.writer.add_scalar(
                f'{phase}/emotion_precision', metrics['emotion_precision'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy', metrics['emotion_accuracy'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/sentiment_precision', metrics['sentiment_precision'], self.global_step)
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy', metrics['sentiment_accuracy'], self.global_step)
            

    
            
       

    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

                # Zero gradient
            self.optimizer.zero_grad()

                # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)
##########################################################################
                # Compute losses using raw logits
            emotion_loss = self.emotion_criterion(outputs['emotion_output'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiment_output'], sentiment_labels)
            total_loss = emotion_loss + sentiment_loss

                # Backward pass
            total_loss.backward()

                # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

                # Tack losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })

            self.global_step += 1

        return {k: v/len(self.train_loader) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase = 'val'):
        self.model.eval()
        losses = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}
        all_emotions_preds = []
        all_sentiments_preds = []
        all_emotions_labels = []
        all_sentiments_labels = []
        
        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                    # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Compute losses
                emotion_loss = self.emotion_criterion(outputs['emotion_output'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment_output'], sentiment_labels)
                total_loss = emotion_loss + sentiment_loss
######################################################################
                    # Collect predictions and labels
                all_emotions_preds.extend(outputs['emotion_output'].argmax(dim=1).cpu().numpy())
                all_sentiments_preds.extend(outputs['sentiment_output'].argmax(dim=1).cpu().numpy())
                all_emotions_labels.extend(emotion_labels.cpu().numpy())
                all_sentiments_labels.extend(sentiment_labels.cpu().numpy())

                    # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()

            # Average losses
        avg_loss = {k: v/len(data_loader) for k, v in losses.items()}

            # compute the precision and accuracy
        emotion_precision = precision_score(
            all_emotions_labels, 
            all_emotions_preds, 
            average='weighted'
        )

        emotion_accuracy = accuracy_score(
            all_emotions_labels, 
            all_emotions_preds
        )

        sentiment_precision = precision_score(
            all_sentiments_labels, 
            all_sentiments_preds, 
            average='weighted'
        )
        sentiment_accuracy = accuracy_score(
            all_sentiments_labels, 
            all_sentiments_preds
        )

        self.log_metrics(avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }, phase = phase)

        if phase == 'val':
            # Step the scheduler with the average total loss
            self.scheduler.step(avg_loss['total'])
            
        return avg_loss, {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy
        }





            



