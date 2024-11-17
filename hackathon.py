from inference_sdk import InferenceHTTPClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="WFIbKZhlG2t0OkPoFFDm"
)

# Define the base directory with subfolders representing building locations
base_dir = "C:/Users/saket/OneDrive/Desktop/hack_disaster/Ukraine locations/"

# Cost estimates for each damage category (min, max)
cost_estimates = {
    "Broken_Window": (200, 800),
    "Roof_Damage": (1500, 9000),
    "Crack_Damage": (500, 5000),
    "Minor_Damage": (100, 500),
    "Major_Damage": (5000, 20000)
}

# Iterate over each subfolder (each representing a unique location)
for location_folder in os.listdir(base_dir):
    location_path = os.path.join(base_dir, location_folder)
    if not os.path.isdir(location_path):
        continue

    # Initialize damage counts and cost totals
    damage_count = {key: 0 for key in cost_estimates.keys()}
    min_total_cost, max_total_cost = 0, 0

    # Process each image in the location folder
    for img_name in os.listdir(location_path):
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            img_path = os.path.join(location_path, img_name)
            
            # Run inference on the image
            result = CLIENT.infer(img_path, model_id="building-damage-dlnea/2")

            # Process predictions
            for pred in result["predictions"]:
                damage_type = pred["class"]
                if damage_type in damage_count:
                    damage_count[damage_type] += 1
                    # Add to the cost totals
                    min_total_cost += cost_estimates[damage_type][0]
                    max_total_cost += cost_estimates[damage_type][1]

    # Generate PDF report for the current location
    report_path = os.path.join(base_dir, f"Report_{location_folder}.pdf")
    c = canvas.Canvas(report_path, pagesize=A4)
    c.setTitle(f"Building Damage Assessment Report for {location_folder}")

    # Header
    c.drawString(50, 800, "Building Damage Assessment Report")
    c.drawString(50, 780, f"Location: {location_folder}")

    # Date of report generation
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(50, 760, f"Report Generated on: {report_date}")

    # Damage Summary Table
    y_position = 720
    c.drawString(50, y_position, "-----------------------------------")
    y_position -= 20
    c.drawString(50, y_position, "| Category       | Count | Estimated Cost (if available) |")
    y_position -= 20
    c.drawString(50, y_position, "-----------------------------------")
    y_position -= 20

    for damage_type, count in damage_count.items():
        if count > 0:
            min_cost, max_cost = cost_estimates[damage_type]
            estimated_cost = f"${min_cost * count} - ${max_cost * count}"
            c.drawString(50, y_position, f"| {damage_type.ljust(15)} | {str(count).ljust(5)} | {estimated_cost.ljust(30)} |")
            y_position -= 20

    # Total Cost Summary
    y_position -= 20
    c.drawString(50, y_position, "-----------------------------------")
    y_position -= 20
    c.drawString(50, y_position, f"Total Estimated Cost: ${min_total_cost} - ${max_total_cost}")

    # Finalize and save the PDF
    c.showPage()
    c.save()

print("Reports generated successfully!")
