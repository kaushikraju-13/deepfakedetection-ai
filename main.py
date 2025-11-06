'''
Main execution script for Deepfake Detection System
Authors: Nischay Upadhya P, Supreeth Gutti, Kaushik Raju S, Nandeesha B
V Semester Mini Project (2025-26)
'''

import argparse
import numpy as np
import sys

from config import Config
from preprocessing import DatasetBuilder
from train import DeepfakeTrainer
from evaluate import ModelEvaluator
from visualize import Visualizer
from inference import DeepfakeDetector
from utils import print_banner, save_json, create_timestamp


def train_model(args):
    '''Complete training pipeline'''
    print_banner("DEEPFAKE DETECTION - TRAINING PIPELINE")
    
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Update config from args
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.img_size:
        config.IMG_SIZE = args.img_size
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Image Size: {config.IMG_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    
    # Step 1: Load Dataset
    print("\n[STEP 1/6] Loading Dataset...")
    trainer = DeepfakeTrainer(config)

    train_gen, val_gen, test_gen = trainer.load_full_dataset()

    # Step 3: Train Model
    print("\n[STEP 3/6] Training Model...")
    trainer.train(
      train_gen,
      None,  # y_train not needed for generators
      val_gen,
      None,
      use_augmentation=False  # we’re already using ImageDataGenerator
    )

    trainer.evaluate_on_test(test_gen, None)
    
    # Step 4: Evaluate Model
    print("\n[STEP 4/6] Evaluating Model...")
    evaluator = ModelEvaluator(trainer.model, config)
    results = evaluator.evaluate(X_test, y_test)
    
    # Step 5: Visualize Results
    print("\n[STEP 5/6] Generating Visualizations...")
    viz = Visualizer(config)
    viz.plot_all_visualizations(
        history, y_test, results['predictions'], 
        results['probabilities'], X_test, results['confusion_matrix']
    )
    
    # Step 6: Save Everything
    print("\n[STEP 6/6] Saving Model and Results...")
    timestamp = create_timestamp()
    trainer.save_model(f'{args.model}_{timestamp}.h5')
    evaluator.save_results(results, f'evaluation_{timestamp}.json')
    
    print_banner("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Final Test AUC: {results['metrics']['auc']:.4f}")
    
    return trainer, results


def test_model(args):
    '''Test trained model on new data'''
    print_banner("DEEPFAKE DETECTION - INFERENCE")
    
    # Load detector
    detector = DeepfakeDetector(args.model_path)
    
    # Single file prediction
    if args.file_path:
        print(f"Analyzing: {args.file_path}")
        
        if args.file_path.endswith(('.mp4', '.avi', '.mov')):
            result = detector.predict_video(args.file_path, threshold=args.threshold)
        else:
            result = detector.predict_image(args.file_path, threshold=args.threshold)
        
        # Display results
        print("\n" + "=" * 80)
        print("DETECTION RESULTS")
        print("=" * 80)
        
        if result['success']:
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probability REAL: {result['probability_real']:.2%}")
            print(f"Probability FAKE: {result['probability_fake']:.2%}")
            if 'num_frames_analyzed' in result:
                print(f"Frames Analyzed: {result['num_frames_analyzed']}")
        else:
            print(f"Error: {result['error']}")
    
    # Batch prediction
    elif args.batch_file:
        with open(args.batch_file, 'r') as f:
            file_paths = [line.strip() for line in f.readlines()]
        
        print(f"Processing {len(file_paths)} files...")
        results = detector.predict_batch(file_paths, threshold=args.threshold)
        
        # Save results
        output_file = f'batch_results_{create_timestamp()}.json'
        save_json(results, output_file)
        
        # Print summary
        fake_count = sum(1 for r in results if r.get('label') == 'FAKE')
        print(f"\nResults: {fake_count} FAKE, {len(results) - fake_count} REAL")
    
    else:
        print("Error: Please provide --file_path or --batch_file")


def main():
    '''Main function'''
    parser = argparse.ArgumentParser(
        description='Deepfake Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train a model
  python main.py train --model efficientnet --epochs 30 --data_type images
  
  # Test on single file
  python main.py test --model_path models/best_model.h5 --file_path test.jpg
  
  # Batch testing
  python main.py test --model_path models/best_model.h5 --batch_file files.txt
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model', type=str, default='efficientnet',
                             choices=['efficientnet', 'xception', 'custom_cnn'],
                             help='Model architecture')
    train_parser.add_argument('--data_type', type=str, default='images',
                             choices=['images', 'videos'],
                             help='Type of dataset')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--img_size', type=int, help='Image size')
    train_parser.add_argument('--video_limit', type=int, help='Limit videos to process')
    train_parser.add_argument('--no-augmentation', dest='augmentation', 
                             action='store_false', help='Disable data augmentation')
    train_parser.set_defaults(augmentation=True)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test trained model')
    test_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained model')
    test_parser.add_argument('--file_path', type=str, help='Single file to test')
    test_parser.add_argument('--batch_file', type=str, help='File with list of paths')
    test_parser.add_argument('--threshold', type=float, default=0.5,
                            help='Classification threshold')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
