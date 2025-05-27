import os
import subprocess

print("[1] Running: KLIPS Sequence Builder")
subprocess.run(["python", "processing/klips_sequence_builder.py"])
print("\n")

print("[2] Running: Transformer AutoEncoder training and Embedding")
subprocess.run(["python", "processing/transformer_autoencoder.py"])
subprocess.run(["python", "processing/transformer_klips.py"])
print("\n")

print("[3] Running: CENSUS Embed Matcher")
subprocess.run(["python", "processing/census_embed_matcher.py"])
print("\n")

print("[4] Running: Selection Variable Generator")
subprocess.run(["python", "selection/make_selection.py"])
subprocess.run(["python", "normalizer/data_normalizer.py"])
print("\n")


print("[5] Running: C2GAM Master Training")
subprocess.run(["python", "C2GAM_training/main.py"])
print("\n")

print("[6] Running: Data inverse scaling")
subprocess.run(["python", "normalizer/data_inverse_normalizer.py"])

print("[7] Plotting the Result in 3D coord")
subprocess.run(["python", "normalizer/plot.py"])

print("\n All steps completed.")
