"""
Extract weights from stage models and print them in Kotlin format for Android
"""

import joblib

print("=" * 70)
print("EXTRACTING MODEL WEIGHTS FOR ANDROID")
print("=" * 70)

model1 = joblib.load("app/stage1_model.pkl")
model2 = joblib.load("app/stage2_model.pkl")
model3 = joblib.load("app/stage3_model.pkl")

print("\n" + "=" * 70)
print("STAGE 1 MODEL (Previous Complications, Preexisting Diabetes, etc.)")
print("=" * 70)
print("Input features: ['Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health']")
print(f"Coefficients: {model1.coef_[0].tolist()}")
print(f"Intercept: {model1.intercept_[0]}")

coef1 = model1.coef_[0].tolist()
intercept1 = model1.intercept_[0]

kotlin_code1 = f"""
    val weight1ofmodel1 = {coef1[0]}
    val weight2ofmodel1 = {coef1[1]}
    val weight3ofmodel1 = {coef1[2]}
    val weight4ofmodel1 = {coef1[3]}
    val biasingofmodel1 = {intercept1}
"""

print("\nKotlin code for Stage 1:")
print(kotlin_code1)

print("\n" + "=" * 70)
print("STAGE 2 MODEL (Abnormality flags)")
print("=" * 70)
print("Input features: ['is_low_bmi', 'is_high_bmi', 'is_low_bp', 'is_high_bp', 'is_high_bs', 'is_high_hr', 'is_low_hr']")
print(f"Coefficients: {model2.coef_[0].tolist()}")
print(f"Intercept: {model2.intercept_[0]}")

coef2 = model2.coef_[0].tolist()
intercept2 = model2.intercept_[0]

kotlin_code2 = f"""
    val weight1ofmodel2 = {coef2[0]}
    val weight2ofmodel2 = {coef2[1]}
    val weight3ofmodel2 = {coef2[2]}
    val weight4ofmodel2 = {coef2[3]}
    val weight5ofmodel2 = {coef2[4]}
    val weight6ofmodel2 = {coef2[5]}
    val weight7ofmodel2 = {coef2[6]}
    val biasingofmodel2 = {intercept2}
"""

print("\nKotlin code for Stage 2:")
print(kotlin_code2)

print("\n" + "=" * 70)
print("STAGE 3 MODEL (Final ensemble)")
print("=" * 70)
print("Input features: ['Risk Score', 'Risk Score Abn']")
print(f"Coefficients: {model3.coef_[0].tolist()}")
print(f"Intercept: {model3.intercept_[0]}")

coef3 = model3.coef_[0].tolist()
intercept3 = model3.intercept_[0]

kotlin_code3 = f"""
    val weight1ofmodel3 = {coef3[0]}
    val weight2ofmodel3 = {coef3[1]}
    val biasingofmodel3 = {intercept3}
"""

print("\nKotlin code for Stage 3:")
print(kotlin_code3)

print("\n" + "=" * 70)
print("COMPLETE KOTLIN CODE TO REPLACE IN Models123.kt")
print("=" * 70)
print(kotlin_code1)
print(kotlin_code2)
print(kotlin_code3)
