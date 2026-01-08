import cv2
import numpy as np

def create_fur_texture(size=(200, 200)):
    # Fur has high frequency noise and edges
    img = np.zeros(size, dtype=np.uint8)
    # Add strong noise
    noise = np.random.randint(0, 255, size, dtype=np.uint8)
    img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)
    # Add some "hairs" (lines)
    for _ in range(50):
        x1, y1 = np.random.randint(0, 200), np.random.randint(0, 200)
        x2, y2 = x1 + np.random.randint(-10, 10), y1 + np.random.randint(-10, 10)
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    return img

def create_skin_texture(size=(200, 200)):
    # Skin is smooth, low frequency changes
    img = np.zeros(size, dtype=np.uint8)
    # Gradient
    for i in range(200):
        val = 150 + i//4
        img[:, i] = val
    # Slight smooth noise
    noise = np.random.normal(0, 5, size).astype(np.uint8)
    img = cv2.add(img, noise)
    # Blur it to make it "skin-like"
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def analyze_crop(name, img):
    # The Algorithm: Variance of Laplacian
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    print(f"--- Analysis: {name} ---")
    print(f"Texture Score: {laplacian_var:.2f}")
    if laplacian_var > 500:
        print("Result: FUR DETECTED (Gato Normal)")
    else:
        print("Result: SKIN DETECTED (Gato Egipcio)")
    print("")
    
    # Save visuals
    cv2.imwrite(f"demo_{name}.original.jpg", img)

if __name__ == "__main__":
    print("Running Texture Feasibility Demo...\n")
    
    fur_img = create_fur_texture()
    skin_img = create_skin_texture()
    
    analyze_crop("Pelaje_Gato", fur_img)
    analyze_crop("Piel_Egipcio", skin_img)
