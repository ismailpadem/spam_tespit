import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import json
import datetime
from pathlib import Path


class SpamDetectionSystem:
    """Complete Spam Detection System with GPU support"""
    
    def __init__(self, max_len=20, embedding_dim=64, learning_rate=0.001):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        # CUDA GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.vocab = {}
        self.vocab_size = 0
        self.model = None
        self.results = {
            'training_start_time': None,
            'training_end_time': None,
            'training_duration': None,
            'dataset_info': {},
            'model_config': {},
            'training_history': {},
            'test_results': {},
            'detailed_metrics': {}
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        print("📚 Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Create vocabulary with minimum frequency threshold
        self.vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
        vocab_idx = 2
        
        for word, count in word_counts.items():
            if count >= min_freq:
                self.vocab[word] = vocab_idx
                vocab_idx += 1
        
        self.vocab_size = len(self.vocab)
        print(f"📖 Vocabulary size: {self.vocab_size:,}")
        print(f"🔤 Most common words: {list(word_counts.most_common(10))}")
        
    def encode_text(self, text):
        """Convert text to numerical sequence"""
        words = text.split()
        encoded = [self.vocab.get(word, 1) for word in words]  # 1 for <UNK>
        return encoded
    
    def pad_sequence(self, seq, max_len):
        """Pad sequence to fixed length"""
        if len(seq) < max_len:
            return seq + [0] * (max_len - len(seq))  # 0 for <PAD>
        else:
            return seq[:max_len]
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the spam dataset"""
        print("📊 Loading and preprocessing data...")
        
        try:
            # Try to load the data
            if not Path(csv_path).exists():
                print(f"❌ File {csv_path} not found!")
                print("Creating sample data for demonstration...")
                # Create sample data if file doesn't exist
                sample_data = {
                    'text': [
                        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
                        "Hello, how are you doing today?",
                        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
                        "I'll meet you at the library at 5pm",
                        "Congratulations! You've won a FREE holiday to Europe!",
                        "Thanks for the information, see you tomorrow",
                        "URGENT! You have won a 1 week FREE membership",
                        "Can you pick up some milk on your way home?",
                        "Call the number below to claim your prize NOW!",
                        "Meeting postponed to next week",
                        "FREE entry to our £2000 prize draw! Text WIN to 85233",
                        "Hope you have a great day!",
                        "Click here to claim your FREE iPhone now!",
                        "Dinner at 7pm tonight?",
                        "WINNER! £1000 cash prize waiting for you!",
                        "Send me the report when you're ready"
                    ] * 100,  # Multiply to get more samples
                    'labels': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100
                }
                df = pd.DataFrame(sample_data)
                print(f"✅ Created sample dataset with {len(df)} samples")
            else:
                # Try different encoding formats and separators
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                separators = [',', '\t', ';', '|']
                
                df = None
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
                            print(f"✅ Loaded dataset with {len(df)} samples using encoding='{encoding}', separator='{sep}'")
                            break
                        except:
                            continue
                    if df is not None:
                        break
                
                if df is None:
                    raise Exception("Could not read CSV file with any encoding/separator combination")
                
                print(f"📋 Dataset columns: {list(df.columns)}")
                print(f"📏 Dataset shape: {df.shape}")
                print(f"🔍 First 3 rows:")
                print(df.head(3))
                
                # Handle different possible column names and formats
                text_col = None
                label_col = None
                
                # Try to identify text column
                text_cols = ['text', 'message', 'content', 'email', 'sms', 'v2', 'body']
                for col in text_cols:
                    if col in df.columns:
                        text_col = col
                        break
                
                # Try to identify label column  
                label_cols = ['labels', 'label', 'spam', 'class', 'category', 'v1', 'target']
                for col in label_cols:
                    if col in df.columns:
                        label_col = col
                        break
                
                # If we can't find obvious columns, use the first two columns
                if text_col is None or label_col is None:
                    if len(df.columns) >= 2:
                        if text_col is None:
                            text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        if label_col is None:
                            label_col = df.columns[0]
                        print(f"🔄 Using columns: text='{text_col}', labels='{label_col}'")
                    else:
                        print(f"❌ Could not identify text and label columns in: {list(df.columns)}")
                        return None
                
                # Extract the relevant columns
                df = df[[label_col, text_col]].copy()
                df.columns = ['labels', 'text']
                
                # Remove any rows with missing values
                df = df.dropna()
                print(f"📊 After removing NaN values: {len(df)} samples")
            
            # Store dataset info
            self.results['dataset_info'] = {
                'total_samples': len(df),
                'csv_path': csv_path
            }
            
            # Clean text data
            print("🧹 Cleaning text data...")
            df['clean_text'] = df['text'].apply(self.clean_text)
            
            # Remove empty texts
            initial_count = len(df)
            df = df[df['clean_text'].str.len() > 0]
            print(f"📊 Removed {initial_count - len(df)} empty texts, remaining: {len(df)}")
            
            # Convert labels to binary (handle both string and numeric labels)
            print("🏷️ Processing labels...")
            if df['labels'].dtype == 'object':
                # String labels like 'spam', 'ham'
                unique_labels = df['labels'].unique()
                print(f"🔤 Found string labels: {unique_labels}")
                
                # Create mapping
                if 'spam' in unique_labels and 'ham' in unique_labels:
                    label_map = {'ham': 0, 'spam': 1}
                else:
                    # Use first unique value as 0, second as 1
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1 if len(unique_labels) > 1 else 0}
                
                df['label_num'] = df['labels'].map(label_map)
                print(f"📊 Label mapping: {label_map}")
            else:
                # Numeric labels - ensure they are 0 and 1
                unique_labels = df['labels'].unique()
                print(f"🔢 Found numeric labels: {unique_labels}")
                df['label_num'] = df['labels'].astype(int)
            
            # Update dataset info with label distribution
            spam_count = sum(df['label_num'])
            ham_count = len(df) - spam_count
            
            self.results['dataset_info'].update({
                'total_samples': len(df),
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_ratio': spam_count / len(df)
            })
            
            # Build vocabulary
            self.build_vocabulary(df['clean_text'].tolist())
            
            # Encode texts
            print("🔤 Encoding texts...")
            df['encoded'] = df['clean_text'].apply(self.encode_text)
            
            # Pad sequences
            print("📏 Padding sequences...")
            df['padded'] = df['encoded'].apply(lambda x: self.pad_sequence(x, self.max_len))
            
            print(f"📈 Final dataset statistics:")
            print(f"   - Total samples: {len(df):,}")
            print(f"   - Spam samples: {spam_count:,}")
            print(f"   - Ham samples: {ham_count:,}")
            print(f"   - Spam ratio: {spam_count / len(df):.2%}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_datasets(self, df, test_size=0.2, random_state=42):
        """Create train and test datasets"""
        print("🔄 Creating train/test split...")
        
        X = df['padded'].tolist()
        y = df['label_num'].tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Create datasets
        train_dataset = SpamDataset(X_train, y_train)
        test_dataset = SpamDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"📊 Train samples: {len(X_train):,}")
        print(f"📊 Test samples: {len(X_test):,}")
        
        return train_loader, test_loader
    
    def create_model(self):
        """Create and initialize the spam classification model"""
        print("🧠 Creating model...")
        
        self.model = SpamClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.results['model_config'] = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_len,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
        
        print(f"🔧 Model parameters: {total_params:,} ({trainable_params:,} trainable)")
        print(f"🚀 Model moved to: {self.device}")
        
        return self.model
    
    def train_model(self, train_loader, test_loader, epochs=10):
        """Train the spam detection model"""
        print(f"🏋️ Starting training for {epochs} epochs...")
        
        # Record training start time
        self.results['training_start_time'] = datetime.datetime.now().isoformat()
        training_start_time = time.time()
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == targets).sum().item()
                train_total += targets.size(0)
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    test_correct += (predictions == targets).sum().item()
                    test_total += targets.size(0)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                  f'Time: {epoch_time:.2f}s')
        
        # Record training completion
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        self.results['training_end_time'] = datetime.datetime.now().isoformat()
        self.results['training_duration'] = total_training_time
        self.results['training_history'] = {
            'epochs': epochs,
            'total_training_time_seconds': total_training_time,
            'total_training_time_formatted': str(datetime.timedelta(seconds=int(total_training_time))),
            'average_epoch_time_seconds': np.mean(epoch_times),
            'best_train_acc': max(history['train_acc']),
            'best_test_acc': max(history['test_acc']),
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_test_loss': history['test_loss'][-1],
            'final_test_acc': history['test_acc'][-1],
            'history': history
        }
        
        print(f"\n✅ Training completed in {datetime.timedelta(seconds=int(total_training_time))}")
        print(f"🎯 Best test accuracy: {max(history['test_acc']):.4f}")
        
        return history
    
    def evaluate_model(self, test_loader):
        """Evaluate the model and calculate detailed metrics"""
        print("📈 Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_targets, all_predictions,
            target_names=['Ham', 'Spam'],
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Store results
        self.results['test_results'] = {
            'accuracy': accuracy,
            'total_samples': len(all_targets),
            'correct_predictions': int(sum(np.array(all_predictions) == np.array(all_targets))),
            'ham_precision': class_report['Ham']['precision'],
            'ham_recall': class_report['Ham']['recall'],
            'ham_f1': class_report['Ham']['f1-score'],
            'spam_precision': class_report['Spam']['precision'],
            'spam_recall': class_report['Spam']['recall'],
            'spam_f1': class_report['Spam']['f1-score'],
            'macro_avg_precision': class_report['macro avg']['precision'],
            'macro_avg_recall': class_report['macro avg']['recall'],
            'macro_avg_f1': class_report['macro avg']['f1-score'],
            'weighted_avg_precision': class_report['weighted avg']['precision'],
            'weighted_avg_recall': class_report['weighted avg']['recall'],
            'weighted_avg_f1': class_report['weighted avg']['f1-score']
        }
        
        self.results['confusion_matrix'] = conf_matrix.tolist()
        
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=['Ham', 'Spam']))
        
        return all_predictions, all_targets
    
    def plot_training_history(self, history, save_path='training_plots.png'):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['test_loss'], label='Test Loss', color='red')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
        plt.plot(history['test_acc'], label='Test Accuracy', color='red')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, predictions, targets, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
        plt.show()
    
    def predict_text(self, text):
        """Predict if a text is spam or ham"""
        self.model.eval()
        
        # Clean and preprocess
        clean = self.clean_text(text)
        if not clean:
            return "ham", 0.0
        
        # Encode
        encoded = self.encode_text(clean)
        
        # Pad
        padded = self.pad_sequence(encoded, self.max_len)
        
        # Convert to tensor
        input_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            prediction = "spam" if probability > 0.5 else "ham"
        
        return prediction, probability
    
    def save_model(self, model_path='spam_model.pth'):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device)
        }, model_path)
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path='spam_model.pth'):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        self.vocab_size = checkpoint['vocab_size']
        self.max_len = checkpoint['max_len']
        self.embedding_dim = checkpoint['embedding_dim']
        
        self.model = SpamClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 Model loaded from {model_path}")
    
    def save_results(self, results_dir='spam_results'):
        """Save comprehensive results"""
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = results_path / f'spam_results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Results saved to {results_file}")
        return results_path


class SpamDataset(Dataset):
    """PyTorch Dataset for spam detection"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpamClassifier(nn.Module):
    """Spam classification model"""
    
    def __init__(self, vocab_size, embedding_dim=64):
        super(SpamClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Fully connected layer
        self.fc = nn.Linear(embedding_dim, 1)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Get embeddings
        embeds = self.embedding(x)
        
        # Average pooling
        pooled = embeds.mean(dim=1)
        
        # Apply dropout
        dropped = self.dropout(pooled)
        
        # Fully connected layer
        out = self.fc(dropped)
        
        # Apply sigmoid
        return self.sigmoid(out).squeeze(1)


def main():
    """Main function to run spam detection training"""
    print("🚀 Starting Spam Detection Training with GPU Support")
    print("=" * 60)
    
    # Initialize system
    spam_detector = SpamDetectionSystem(
        max_len=20,
        embedding_dim=128,  # Increased for better performance
        learning_rate=0.001
    )
    
    # Load and preprocess data - Updated path to use your dataset
    csv_path = "data/spam.csv"  # Updated path
    print(f"📁 Loading dataset from: {csv_path}")
    df = spam_detector.load_and_preprocess_data(csv_path)
    
    if df is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Create datasets
    train_loader, test_loader = spam_detector.create_datasets(df)
    
    # Create model
    model = spam_detector.create_model()
    
    # Train model
    history = spam_detector.train_model(train_loader, test_loader, epochs=10)
    
    # Evaluate model
    predictions, targets = spam_detector.evaluate_model(test_loader)
    
    # Plot results
    spam_detector.plot_training_history(history, save_path='training_plots.png')
    spam_detector.plot_confusion_matrix(predictions, targets, save_path='confusion_matrix.png')
    
    # Save model and results
    spam_detector.save_model('spam_model_gpu.pth')
    spam_detector.save_results('spam_results')
    
    # Test predictions
    print("\n🧪 Testing predictions:")
    test_messages = [
        "Free entry! Win a brand new iPhone now!",
        "I'll meet you at the library at 5pm",
        "Congratulations! You've won a FREE holiday!",
        "Can you pick up some milk on your way home?",
        "URGENT! Click here to claim your prize NOW!",
        "Thanks for the information, see you tomorrow"
    ]
    
    for message in test_messages:
        prediction, probability = spam_detector.predict_text(message)
        print(f"'{message[:50]}...' -> {prediction.upper()} ({probability:.3f})")
    
    # Display final results summary
    print(f"\n✅ Spam detection training completed successfully!")
    print(f"🎯 Final Results:")
    print(f"   - Test Accuracy: {spam_detector.results['test_results']['accuracy']:.4f}")
    print(f"   - Training Time: {spam_detector.results['training_history']['total_training_time_formatted']}")
    print(f"   - Device Used: {spam_detector.device}")
    print(f"   - Dataset: {csv_path}")
    print(f"   - Total Samples: {spam_detector.results['dataset_info']['total_samples']:,}")
    print(f"   - Model Parameters: {spam_detector.results['model_config']['total_parameters']:,}")
    
    # Interactive testing (optional)
    user_input = input("\n🎮 Would you like to test custom messages? (y/n): ")
    if user_input.lower() == 'y':
        print("Interactive testing (type 'quit' to exit):")
        while True:
            message = input("Enter a message: ")
            if message.lower() == 'quit':
                break
            
            prediction, probability = spam_detector.predict_text(message)
            print(f"Prediction: {prediction.upper()} (Confidence: {probability:.3f})")
    
    print("\n🎉 All done! Check the spam_tespit folder for saved results and plots.")


if __name__ == "__main__":
    main() 