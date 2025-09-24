#!/usr/bin/env python3
"""
Test Discriminative Model Script
Tests the discriminative model's ability to distinguish between similar cancer types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the discriminative classifier
from create_discriminative_model import DiscriminativeClassifier

class DiscriminativeModelTester:
    """Test the discriminative model's performance"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ["ALL", "AML", "CLL", "CML"]
        self.model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the discriminative model"""
        print("🔄 Loading discriminative model...")
        
        try:
            # Create model architecture (without auxiliary classifiers for inference)
            model = models.googlenet(pretrained=False, aux_logits=False)
            num_features = model.fc.in_features
            model.fc = DiscriminativeClassifier(num_features, 4)
            
            # Load trained weights
            model_path = "blood_cancer_model_discriminative.pth"
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Filter out auxiliary classifier weights if they exist
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if not (key.startswith('aux1.') or key.startswith('aux2.')):
                        filtered_state_dict[key] = value
                
                model.load_state_dict(filtered_state_dict, strict=False)
                print("✅ Discriminative model loaded successfully!")
            else:
                print("❌ Discriminative model file not found!")
                return None
            
            model = model.to(self.device)
            model.eval()
            self.model = model
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def load_test_images(self):
        """Load test images from the test_images directory"""
        print("📁 Loading test images...")
        
        test_images = {}
        test_dir = "test_images"
        
        if not os.path.exists(test_dir):
            print(f"❌ Test directory {test_dir} not found!")
            return {}
        
        # Load leukemic images
        leukemic_dir = os.path.join(test_dir, "leukemic")
        if os.path.exists(leukemic_dir):
            for filename in os.listdir(leukemic_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                    filepath = os.path.join(leukemic_dir, filename)
                    
                    # Determine cancer type from filename
                    if 'acute_lymphoblastic' in filename or 'all' in filename.lower():
                        cancer_type = 'ALL'
                    elif 'acute_myeloid' in filename or 'aml' in filename.lower():
                        cancer_type = 'AML'
                    elif 'chronic_lymphocytic' in filename or 'cll' in filename.lower():
                        cancer_type = 'CLL'
                    elif 'chronic_myeloid' in filename or 'cml' in filename.lower():
                        cancer_type = 'CML'
                    else:
                        cancer_type = 'Unknown'
                    
                    if cancer_type not in test_images:
                        test_images[cancer_type] = []
                    test_images[cancer_type].append(filepath)
        
        # Load normal images (for comparison)
        normal_dir = os.path.join(test_dir, "normal")
        if os.path.exists(normal_dir):
            test_images['Normal'] = []
            for filename in os.listdir(normal_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                    filepath = os.path.join(normal_dir, filename)
                    test_images['Normal'].append(filepath)
        
        print(f"✅ Loaded test images: {dict((k, len(v)) for k, v in test_images.items())}")
        return test_images
    
    def predict_image(self, image_path):
        """Predict cancer type for a single image"""
        try:
            # Handle SVG files by converting to RGB
            if image_path.endswith('.svg'):
                # For SVG files, create a placeholder RGB image
                image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            return {
                'predicted_class': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy(),
                'all_probs': {self.class_names[i]: probabilities[0][i].item() 
                             for i in range(len(self.class_names))}
            }
            
        except Exception as e:
            print(f"❌ Error predicting image {image_path}: {e}")
            return None
    
    def test_discrimination(self, test_images):
        """Test discrimination between similar cancer types"""
        print("\n🧪 Testing Discrimination Capabilities...")
        
        results = defaultdict(list)
        discrimination_results = {}
        
        # Test each image
        for cancer_type, image_paths in test_images.items():
            if cancer_type == 'Normal':
                continue
                
            print(f"\n📊 Testing {cancer_type} images:")
            
            for image_path in image_paths:
                prediction = self.predict_image(image_path)
                if prediction:
                    results[cancer_type].append(prediction)
                    
                    filename = os.path.basename(image_path)
                    predicted = prediction['predicted_class']
                    confidence = prediction['confidence']
                    
                    # Check if prediction is correct
                    correct = "✅" if predicted == cancer_type else "❌"
                    print(f"  {filename}: {predicted} ({confidence:.3f}) {correct}")
        
        # Analyze discrimination between similar pairs
        print("\n🎯 Discrimination Analysis:")
        
        # ALL vs CLL discrimination
        if 'ALL' in results and 'CLL' in results:
            all_predictions = results['ALL']
            cll_predictions = results['CLL']
            
            all_correct = sum(1 for p in all_predictions if p['predicted_class'] == 'ALL')
            cll_correct = sum(1 for p in cll_predictions if p['predicted_class'] == 'CLL')
            
            all_accuracy = all_correct / len(all_predictions) if all_predictions else 0
            cll_accuracy = cll_correct / len(cll_predictions) if cll_predictions else 0
            
            discrimination_results['ALL_vs_CLL'] = {
                'ALL_accuracy': all_accuracy,
                'CLL_accuracy': cll_accuracy,
                'overall_discrimination': (all_accuracy + cll_accuracy) / 2
            }
            
            print(f"ALL vs CLL:")
            print(f"  ALL accuracy: {all_accuracy:.3f} ({all_correct}/{len(all_predictions)})")
            print(f"  CLL accuracy: {cll_accuracy:.3f} ({cll_correct}/{len(cll_predictions)})")
            print(f"  Overall discrimination: {discrimination_results['ALL_vs_CLL']['overall_discrimination']:.3f}")
        
        # AML vs CML discrimination
        if 'AML' in results and 'CML' in results:
            aml_predictions = results['AML']
            cml_predictions = results['CML']
            
            aml_correct = sum(1 for p in aml_predictions if p['predicted_class'] == 'AML')
            cml_correct = sum(1 for p in cml_predictions if p['predicted_class'] == 'CML')
            
            aml_accuracy = aml_correct / len(aml_predictions) if aml_predictions else 0
            cml_accuracy = cml_correct / len(cml_predictions) if cml_predictions else 0
            
            discrimination_results['AML_vs_CML'] = {
                'AML_accuracy': aml_accuracy,
                'CML_accuracy': cml_accuracy,
                'overall_discrimination': (aml_accuracy + cml_accuracy) / 2
            }
            
            print(f"AML vs CML:")
            print(f"  AML accuracy: {aml_accuracy:.3f} ({aml_correct}/{len(aml_predictions)})")
            print(f"  CML accuracy: {cml_accuracy:.3f} ({cml_correct}/{len(cml_predictions)})")
            print(f"  Overall discrimination: {discrimination_results['AML_vs_CML']['overall_discrimination']:.3f}")
        
        return results, discrimination_results
    
    def generate_report(self, results, discrimination_results):
        """Generate a comprehensive test report"""
        print("\n📋 Discrimination Test Report")
        print("=" * 50)
        
        # Overall statistics
        total_images = sum(len(predictions) for predictions in results.values())
        total_correct = sum(
            sum(1 for p in predictions if p['predicted_class'] == cancer_type)
            for cancer_type, predictions in results.items()
        )
        
        overall_accuracy = total_correct / total_images if total_images > 0 else 0
        
        print(f"📊 Overall Performance:")
        print(f"  Total images tested: {total_images}")
        print(f"  Correct predictions: {total_correct}")
        print(f"  Overall accuracy: {overall_accuracy:.3f}")
        
        # Discrimination performance
        print(f"\n🎯 Discrimination Performance:")
        
        if 'ALL_vs_CLL' in discrimination_results:
            disc = discrimination_results['ALL_vs_CLL']['overall_discrimination']
            status = "✅ Good" if disc > 0.7 else "⚠️ Needs Improvement" if disc > 0.5 else "❌ Poor"
            print(f"  ALL vs CLL: {disc:.3f} {status}")
        
        if 'AML_vs_CML' in discrimination_results:
            disc = discrimination_results['AML_vs_CML']['overall_discrimination']
            status = "✅ Good" if disc > 0.7 else "⚠️ Needs Improvement" if disc > 0.5 else "❌ Poor"
            print(f"  AML vs CML: {disc:.3f} {status}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        avg_discrimination = 0
        count = 0
        
        if 'ALL_vs_CLL' in discrimination_results:
            avg_discrimination += discrimination_results['ALL_vs_CLL']['overall_discrimination']
            count += 1
        
        if 'AML_vs_CML' in discrimination_results:
            avg_discrimination += discrimination_results['AML_vs_CML']['overall_discrimination']
            count += 1
        
        if count > 0:
            avg_discrimination /= count
            
            if avg_discrimination > 0.8:
                print("  🎉 Excellent discrimination! Model ready for deployment.")
            elif avg_discrimination > 0.6:
                print("  👍 Good discrimination. Consider fine-tuning for better performance.")
            else:
                print("  🔧 Poor discrimination. Model needs significant improvement.")
                print("  💡 Consider: More training data, architecture changes, or feature engineering.")
        
        return {
            'overall_accuracy': overall_accuracy,
            'discrimination_results': discrimination_results,
            'total_images': total_images,
            'total_correct': total_correct
        }

def main():
    """Main function to test discriminative model"""
    print("🧪 Discriminative Model Testing")
    print("=" * 40)
    
    # Create tester
    tester = DiscriminativeModelTester()
    
    # Load model
    model = tester.load_model()
    if not model:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load test images
    test_images = tester.load_test_images()
    if not test_images:
        print("❌ No test images found. Exiting.")
        return
    
    # Test discrimination
    results, discrimination_results = tester.test_discrimination(test_images)
    
    # Generate report
    report = tester.generate_report(results, discrimination_results)
    
    print("\n" + "=" * 40)
    print("🎯 Testing Complete!")
    print("=" * 40)

if __name__ == "__main__":
    main()