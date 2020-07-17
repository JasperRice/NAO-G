from PIL import Image 
  
# Opens a image in RGB mode
result = "dataset/Figures/Test_Result_"
ground = "dataset/Figures/Test_Result_Ground_"

for i in range(1, 21):
    if i < 10:
        current_result = result + '0'
        current_ground = ground + '0'
    else:
        current_result = result
        current_ground = ground
    current_result += str(i) + '.png'
    current_ground += str(i) + '.png'
    
    current_result_image = Image.open(current_result) 
    current_ground_image = Image.open(current_ground) 
  
    # Size of the image in pixels (size of orginal image) 
    r_width, r_height = current_result_image.size
    g_width, g_height = current_ground_image.size

    # Setting the points for cropped image
    offset = - 100
    left = r_width / 3 + offset
    right = 2 * r_width / 3 + offset
    
    top = 250
    bottom = r_height - 50
    
    # Cropped image of above dimension 
    cropped_current_result_image = current_result_image.crop((left, top, right, bottom)) 
    cropped_current_ground_image = current_ground_image.crop((left, top, right, bottom)) 
    
    # Shows the image in image viewer 
    cropped_current_result_image.save(current_result[:-4]+'_Cropped'+current_result[-4:])
    cropped_current_ground_image.save(current_ground[:-4]+'_Cropped'+current_ground[-4:])