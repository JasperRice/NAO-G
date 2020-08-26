from PIL import Image 
  
human = "dataset/Figures/Cropped_Type_Human_Hand_Wide_"
nao = "dataset/Figures/Cropped_Type_NAO_Hand_Wide_"
save = "dataset/Figures/Type_Hand_Wide_"

for i in range(1, 100):
    if i < 10:
        current_human = human + '0'
        current_nao = nao + '0'
        current_save = save + '0'
    else:
        current_human = human
        current_nao = nao
        current_save = save
    current_human += str(i) + '.png'
    current_nao += str(i) + '.png'
    current_save += str(i) + '.png'

    try: 
        images = [Image.open(x) for x in [current_human, current_nao]]
        widths, heights = zip(*(i.size for i in images))
    except: break

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(current_save)