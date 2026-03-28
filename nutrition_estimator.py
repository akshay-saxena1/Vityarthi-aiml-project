import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. The Nutrition Database (Macros per 100 grams)
NUTRITION_DB = {
    "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fats": 10},
    "hamburger": {"calories": 295, "protein": 17, "carbs": 24, "fats": 14},
    "sushi": {"calories": 143, "protein": 4.5, "carbs": 31, "fats": 0.5},
    "samosa": {"calories": 262, "protein": 3.5, "carbs": 24, "fats": 17},
    "french_fries": {"calories": 312, "protein": 3.4, "carbs": 41, "fats": 15},
    "ice_cream": {"calories": 207, "protein": 3.5, "carbs": 24, "fats": 11},
    # A fallback for any food not listed above
    "default": {"calories": 250, "protein": 10, "carbs": 20, "fats": 10} 
}

# 3. Portion Size Multipliers (Estimating weight in grams)
PORTIONS = {
    "1": {"name": "Small (approx 150g)", "multiplier": 1.5},
    "2": {"name": "Medium (approx 250g)", "multiplier": 2.5},
    "3": {"name": "Large (approx 400g)", "multiplier": 4.0}
}

# 4. Load the Class Names
train_dir = None
for root, dirs, files in os.walk('./dataset'):
    if 'train' in dirs:
        train_dir = os.path.join(root, 'train')
        break

if not train_dir:
    raise FileNotFoundError("Could not find the dataset train folder to load class names!")

class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# 5. Load YOUR 12-Epoch Trained Model
print("Loading High-Performance AI Model...")
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Loading the 'best' model from your max-GPU training run
model.load_state_dict(torch.load('best_food101_model.pth', weights_only=True))
model = model.to(device)
model.eval()

# 6. Image Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 7. The Main Pipeline
def analyze_meal(image_path):
    print(f"\n📸 Analyzing image: {image_path}...")
    
    # --- A. COMPUTER VISION (Identify the food) ---
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_tensor = transform(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    food_name = class_names[predicted_idx.item()]
    print(f"🤖 AI Recognized: {food_name.upper()}")
    
    # --- B. PORTION ESTIMATION (User Input) ---
    print("\n🍽️  How big is the portion?")
    print("1: Small  (~150g)")
    print("2: Medium (~250g)")
    print("3: Large  (~400g)")
    choice = input("Enter 1, 2, or 3: ").strip()
    
    # Default to medium if they type something weird
    if choice not in PORTIONS:
        print("Invalid choice, defaulting to Medium.")
        choice = "2"
        
    multiplier = PORTIONS[choice]["multiplier"]
    
    # --- C. NUTRITION MAPPING ---
    macros = NUTRITION_DB.get(food_name, NUTRITION_DB["default"])
    
    total_cals = round(macros["calories"] * multiplier)
    total_protein = round(macros["protein"] * multiplier, 1)
    total_carbs = round(macros["carbs"] * multiplier, 1)
    total_fats = round(macros["fats"] * multiplier, 1)
    
    # --- D. FINAL OUTPUT ---
    print("\n" + "=" * 40)
    print(f"🔥 NUTRITION ESTIMATE FOR: {food_name.replace('_', ' ').title()}")
    print(f"⚖️  Portion: {PORTIONS[choice]['name']}")
    print("-" * 40)
    print(f"⚡ Calories:      {total_cals} kcal")
    print(f"🥩 Protein:       {total_protein} g")
    print(f"🍞 Carbohydrates: {total_carbs} g")
    print(f"🧈 Fats:          {total_fats} g")
    print("=" * 40 + "\n")

# Run the app!
test_image_path = "test_food.jpg" 
if os.path.exists(test_image_path):
    analyze_meal(test_image_path)
else:
    print(f"\n❌ Error: Could not find '{test_image_path}'.")
    print("Please save a test image in your project folder as 'test_food.jpg' and try again.\n")