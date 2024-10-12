#%%
import pyautogui
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


#%%

# Take a screenshot

# Take a screenshot
screenshot = pyautogui.screenshot()

# Pre-process the screenshot
screenshot_gray = screenshot.convert('L')
screenshot_gray = ImageEnhance.Contrast(screenshot_gray).enhance(2)
screenshot_gray = screenshot_gray.point(lambda x: 0 if x<128 else 255, '1')

# Perform OCR
text = pytesseract.image_to_string(screenshot_gray)

# Find the word and its location
word_to_find = 'Review'
x_global = None
y_global = None

for line in text.splitlines():
    if word_to_find in line:
        # Use enumerate to get the index (y-coordinate) and the line text
        for y, line_text in enumerate(text.splitlines()):
            # Use str.find to get the x-coordinate
            x = line_text.find(word_to_find)
            if x != -1:
                x += 10  # adjust x-coordinate to click on the center of the word
                y += 10  # adjust y-coordinate to click on the center of the word
                x_global = x
                y_global = y
                break
        if x_global is not None and y_global is not None:
            break

# Check if the word was found
if x_global is not None and y_global is not None:
    # Click on the word
    pyautogui.moveTo(x_global, y_global)
    pyautogui.click()
else:
    print(f"Word '{word_to_find}' not found on the screen.")
# %%
