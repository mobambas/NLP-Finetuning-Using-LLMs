import json

# Path to your notebook
notebook_path = "C:\\Users\\shriy\\Downloads\\Meta_Learning_for_Efficient_Fine_Tuning_of_LLMs (1).ipynb"
fixed_path = "C:\\Users\\shriy\\Downloads\\MyNotebook_fixed.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove widgets from metadata if it exists
if "metadata" in data:
    if "widgets" in data["metadata"]:
        del data["metadata"]["widgets"]

# Optional: simplify metadata for GitHub/Colab rendering
# Keep only essential parts
essential_metadata = {
    "colab": data["metadata"].get("colab", {}),
    "kernelspec": data["metadata"].get("kernelspec", {})
}
data["metadata"] = essential_metadata

# Save fixed notebook
with open(fixed_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Fixed notebook saved as {fixed_path}")
