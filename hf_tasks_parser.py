import json
import argparse
import requests
from pathlib import Path
import sys
import io
from collections import defaultdict

# Force UTF-8 encoding on Windows to handle emojis
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Hugging Face Tasks API endpoint
HUGGINGFACE_TASKS_API = "https://huggingface.co/api/tasks"

def fetch_huggingface_tasks():
    try:
        print("🔄 Fetching tasks from Hugging Face API...")
        response = requests.get(HUGGINGFACE_TASKS_API, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print("✅ Successfully fetched data from API")       

        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from API: {e}")
        return None

def save_tasks_to_file(data, filename="huggingface_tasks.json"):
    """
    Save the fetched tasks data to a local JSON file
    
    Args:
        data (dict): Tasks data to save
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"💾 Data saved to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving data to file: {e}")

def categorize_task(task_id):
    """Categorize a task based on its ID to match Hugging Face's official categorization"""
    
    # Multimodal tasks
    multimodal_tasks = {
        'any-to-any', 'audio-text-to-text', 'document-question-answering', 
        'visual-document-retrieval', 'image-text-to-text', 'video-text-to-text', 
        'visual-question-answering'
    }
    
    # Natural Language Processing tasks
    nlp_tasks = {
        'feature-extraction', 'fill-mask', 'question-answering', 'sentence-similarity',
        'summarization', 'table-question-answering', 'text-classification', 
        'text-generation', 'text-ranking', 'token-classification', 'translation',
        'zero-shot-classification'
    }
    
    # Computer Vision tasks  
    cv_tasks = {
        'depth-estimation', 'image-classification', 'image-feature-extraction',
        'image-segmentation', 'image-to-image', 'image-to-text', 'keypoint-detection',
        'mask-generation', 'object-detection', 'video-classification', 'text-to-image',
        'text-to-video', 'unconditional-image-generation', 'zero-shot-image-classification',
        'zero-shot-object-detection', 'text-to-3d', 'image-to-3d', 'image-to-video'
    }
    
    # Audio tasks
    audio_tasks = {
        'audio-classification', 'audio-to-audio', 'automatic-speech-recognition', 
        'text-to-speech'
    }
    
    # Tabular tasks
    tabular_tasks = {
        'tabular-classification', 'tabular-regression'
    }
    
    # Reinforcement Learning tasks
    rl_tasks = {
        'reinforcement-learning'
    }
    
    # Check exact matches first
    if task_id in multimodal_tasks:
        return 'Multimodal'
    elif task_id in nlp_tasks:
        return 'Natural Language Processing'
    elif task_id in cv_tasks:
        return 'Computer Vision'
    elif task_id in audio_tasks:
        return 'Audio'
    elif task_id in tabular_tasks:
        return 'Tabular'
    elif task_id in rl_tasks:
        return 'Reinforcement Learning'
    else:
        return 'Other'

def display_models_only_breakdown(data):
    """
    Display breakdown focusing only on models, organized by category
    """
    print("\n🤖 MODELS-FOCUSED BREAKDOWN")
    print("=" * 80)
    print()
    
    valid_tasks = [(task_id, task_info) for task_id, task_info in data.items() 
                   if not task_info.get('isPlaceholder', False)]
    
    # Organize by category
    categories = defaultdict(list)
    for task_id, task_info in valid_tasks:
        category = categorize_task(task_id)
        categories[category].append((task_id, task_info))
    
    for category in sorted(categories.keys()):
        tasks_in_category = categories[category]
        all_models = []
        
        # Collect all models from all tasks in this category
        for task_id, task_info in tasks_in_category:
            models = task_info.get('models', [])
            for model in models:
                all_models.append({
                    'task': task_info.get('label', task_id),
                    'task_id': task_id,
                    'model_id': model.get('id', 'Unknown'),
                    'description': model.get('description', 'No description')
                })
        
        print(f"🗂️  {category.upper()} - {len(all_models)} Models")
        print("=" * 60)
        print()
        
        if all_models:
            # Group models by task
            task_models = defaultdict(list)
            for model in all_models:
                task_models[model['task']].append(model)
            
            for task_name in sorted(task_models.keys()):
                models_in_task = task_models[task_name]
                # Find the task info for summary and YouTube
                task_summary = ""
                youtube_id = ""
                for model in models_in_task:
                    if task_summary == "":  # Get info from first model's task
                        task_data = next((taskInfo for taskId, taskInfo in tasks_in_category if taskInfo.get('label', taskId) == task_name), None)
                        if task_data:
                            task_summary = task_data.get('summary', 'No summary available')
                            youtube_id = task_data.get('youtubeId', '')
                        break
                
                print(f"📋 {task_name} ({len(models_in_task)} models)")
                print(f"    SUMMARY: {task_summary}")
                if youtube_id:
                    print(f"    📺 YOUTUBE: https://www.youtube.com/watch?v={youtube_id}")
                else:
                    print(f"    📺 YOUTUBE: No video available")
                print()
                
                # Show ALL models (not limited)
                for i, model in enumerate(models_in_task, 1):
                    print(f"   {i}. {model['model_id']}")
                    print(f"      → {model['description']}")
                print()
        else:
            print("   No models available in this category.")
            print()
        
        print()

def display_category_summary(data):
    """Display a summary of categories with task and model counts"""
    
    valid_tasks = [(task_id, task_info) for task_id, task_info in data.items() 
                   if not task_info.get('isPlaceholder', False)]
    
    print("\n📊 CATEGORY SUMMARY")
    print("=" * 50)
    print()
    
    categories = defaultdict(lambda: {'tasks': 0, 'models': 0, 'task_list': []})
    
    for task_id, task_info in valid_tasks:
        category = categorize_task(task_id)
        models_count = len(task_info.get('models', []))
        
        categories[category]['tasks'] += 1
        categories[category]['models'] += models_count
        categories[category]['task_list'].append(task_info.get('label', task_id))
    
    # Display summary table
    print(f"{'Category':<30} {'Tasks':<8} {'Models':<8} {'Avg Models/Task':<15}")
    print("-" * 65)
    
    total_tasks = 0
    total_models = 0
    
    for category in sorted(categories.keys()):
        stats = categories[category]
        avg_models = stats['models'] / stats['tasks'] if stats['tasks'] > 0 else 0
        
        print(f"{category:<30} {stats['tasks']:<8} {stats['models']:<8} {avg_models:<15.1f}")
        
        total_tasks += stats['tasks']
        total_models += stats['models']
    
    print("-" * 65)
    overall_avg = total_models / total_tasks if total_tasks > 0 else 0
    print(f"{'TOTAL':<30} {total_tasks:<8} {total_models:<8} {overall_avg:<15.1f}")
    print()
    
    # Show top categories
    print("🏆 TOP CATEGORIES BY MODEL COUNT:")
    top_cats = sorted(categories.items(), key=lambda x: x[1]['models'], reverse=True)[:5]
    
    for i, (category, stats) in enumerate(top_cats, 1):
        print(f"{i}. {category}: {stats['models']} models across {stats['tasks']} tasks")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Hugging Face Tasks API Parser')
    parser.add_argument('--save', action='store_true',
                       help='Save fetched data to a JSON file')
    parser.add_argument('--file', '-f', type=str,
                       help='Use local JSON file instead of fetching from API')
    parser.add_argument('--models', action='store_true',
                       help='Show models-focused breakdown')
    parser.add_argument('--summary', action='store_true',
                       help='Show category summary only')
    
    args = parser.parse_args()
    
    # Load data from file or API
    if args.file:
        if not Path(args.file).exists():
            print(f"❌ Error: File '{args.file}' not found.")
            return
        
        try:
            print(f"📂 Reading data from '{args.file}'...")
            with open(args.file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print("✅ Successfully loaded data from file")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    else:
        data = fetch_huggingface_tasks()
        if data is None:
            print("❌ Failed to fetch data from API. Exiting.")
            return
        
        if args.save:
            save_tasks_to_file(data)
    
    # Process the data based on arguments
    if args.summary:
        display_category_summary(data)
    else:
        display_category_summary(data)
        display_models_only_breakdown(data)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🤗 Comprehensive Hugging Face Tasks Parser")
        print("=" * 45)
        print()
        print("This script provides complete breakdown of all Hugging Face tasks and models.")
        print()
        print("Usage:")
        print("  python script.py                    # Complete breakdown by category")
        print("  python script.py --models           # Focus on models organization")
        print("  python script.py --summary          # Category summary with statistics")
        print("  python script.py --save             # Fetch and save data to file")
        print("  python script.py --file tasks.json  # Use local JSON file")
        print()
        print("Features:")
        print("✓ Complete task breakdown by category")
        print("✓ All models listed with descriptions") 
        print("✓ Task metadata (libraries, datasets, spaces)")
        print("✓ Statistical summaries and rankings")
        print("✓ Live API fetching with local file fallback")
        print()
        
        # Show quick demo
        print("Fetching quick preview...")
        data = fetch_huggingface_tasks()
        if data:
            display_category_summary(data)
            display_models_only_breakdown(data)
    else:
        main()