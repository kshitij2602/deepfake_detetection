import shutil
import os
import random

src_real = "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/real"
src_fake = "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/fake"

dst = "/kaggle/working/demo_images"
os.makedirs(dst, exist_ok=True)

for cls in ["real", "fake"]:
    src = src_real if cls == "real" else src_fake
    files = random.sample(os.listdir(src), 5)
    for f in files:
        shutil.copy(os.path.join(src, f), dst)
