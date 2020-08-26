from PIL import Image 
  
# Opens a image in RGB mode
test = "dataset/Figures/Type_NAO_Hand_Wide_"
slash = 0
for i, c in enumerate(test):
    if c == '/': slash = i+1

for i in range(1, 100):
    if i < 10: current_test = test + '0'
    else: current_test = test
    current_test += str(i) + '.png'
    try: current_test_image = Image.open(current_test)
    except: break
  
    # Size of the image in pixels (size of orginal image) 
    width, height = current_test_image.size

    # Setting the points for cropped image
    offset = - 75
    left = width / 3 + offset + (0)
    right = 2 * width / 3 + offset - (0)
    
    top = 300
    bottom = height - 50
    
    # Cropped image of above dimension 
    cropped_current_test_image = current_test_image.crop((left, top, right, bottom))
    cropped_current_test_image = cropped_current_test_image.resize((640 * 680//730, 680))
    
    # Shows the image in image viewer 
    cropped_current_test_image.save(current_test[:slash]+'Cropped_'+current_test[slash:])