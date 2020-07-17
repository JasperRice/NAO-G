from PIL import Image 
  
result = "dataset/Figures/Test_Result_"
ground = "dataset/Figures/Test_Result_Ground_"

for i in range(1, 21):
    if i < 10:
        current_result = result + '0'
        current_ground = ground + '0'
    else:
        current_result = result
        current_ground = ground
    current_result += str(i) + '_Cropped.png'
    current_ground += str(i) + '_Cropped.png'

    images = [Image.open(x) for x in [current_result, current_ground]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(current_ground[:-4]+'_Merged'+current_ground[-4:])