import os
import subprocess

print("[1] Running: KLIPS Sequence Builder")
subprocess.run(["python", "transformer_embedder/klips_sequence_builder.py"])
print("\n")

print("[2] Running: Transformer Encoder")
subprocess.run(["python", "transformer_embedder/transformer_encoder.py"])
print("\n")

print("[3] Running: CENSUS Embed Matcher")
subprocess.run(["python", "transformer_embedder/census_embed_matcher.py"])
print("\n")

print("[4] Running: Selection Variable Generator")
subprocess.run(["python", "transformer_embedder/make_selection.py"])
print("\n")

print("[5] Running: C2GAM Master Training")
subprocess.run(["python", "C2GAM_master/main.py"])
print("\n")

print("\n✅ All steps completed.")
