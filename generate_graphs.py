import matplotlib.pyplot as plt
import seaborn as sns

print("Generating Energy & Health Proportion Graphs...")

# Your mock database for the presentation
nutrition_db = {
    "apple_pie": {"healthy": False, "energy_kcal": 300},
    "grilled_salmon": {"healthy": True, "energy_kcal": 250},
    "hamburger": {"healthy": False, "energy_kcal": 500},
    "salad": {"healthy": True, "energy_kcal": 150},
    "pizza": {"healthy": False, "energy_kcal": 285},
    "edamame": {"healthy": True, "energy_kcal": 120}
}

foods = list(nutrition_db.keys())
calories = [nutrition_db[f]["energy_kcal"] for f in foods]
health_status = ["Healthy" if nutrition_db[f]["healthy"] else "Non-Healthy" for f in foods]

healthy_count = health_status.count("Healthy")
unhealthy_count = health_status.count("Non-Healthy")

plt.figure(figsize=(12, 5))

# Plot 1: Pie Chart
plt.subplot(1, 2, 1)
plt.pie([healthy_count, unhealthy_count], labels=["Healthy", "Non-Healthy"], 
        autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
plt.title("Proportion of Healthy vs Non-Healthy Items")

# Plot 2: Bar Chart
plt.subplot(1, 2, 2)
sns.barplot(x=calories, y=foods, palette="viridis")
plt.xlabel("Energy (kcal per serving)")
plt.ylabel("Food Item")
plt.title("Energy Content Profile")

plt.tight_layout()
plt.savefig("nutritional_analysis.png")
print("✅ Graphs saved successfully as 'nutritional_analysis.png' in your folder!")