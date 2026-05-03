"""
Extract StandardScaler parameters and print them in Kotlin format for Android
"""

import joblib

print("=" * 70)
print("EXTRACTING SCALER PARAMETERS FOR ANDROID")
print("=" * 70)

scaler = joblib.load("app/scaler.pkl")

print("\nFeatures:", scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else "Unknown")
print("Feature means (intercept):", scaler.mean_.tolist())
print("Feature stds (scale):", scaler.scale_.tolist())

features = ['BS', 'BMI', 'Age', 'Heart Rate', 'Systolic BP', 'Diastolic']
means = scaler.mean_.tolist()
stds = scaler.scale_.tolist()

print("\n" + "=" * 70)
print("KOTLIN CODE FOR STANDARDIZATION")
print("=" * 70)

kotlin_code = """
// StandardScaler parameters for vitals: [BS, BMI, Age, Heart Rate, Systolic BP, Diastolic]
val scaler_means = doubleArrayOf(
"""

for i, (feat, mean) in enumerate(zip(features, means)):
    kotlin_code += f"    {mean}  // {feat}\n"

kotlin_code += """)

val scaler_scales = doubleArrayOf(
"""

for i, (feat, std) in enumerate(zip(features, stds)):
    kotlin_code += f"    {std}  // {feat}\n"

kotlin_code += """)\

// Standardization function
fun standardizeVitals(vitals: DoubleArray): DoubleArray {
    return DoubleArray(6) { i ->
        (vitals[i] - scaler_means[i]) / scaler_scales[i]
    }
}
"""

print(kotlin_code)

print("\n" + "=" * 70)
print("THESE VALUES SHOULD BE ADDED TO YOUR STANDARDIZATION CODE")
print("=" * 70)
