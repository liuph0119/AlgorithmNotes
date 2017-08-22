# !/usr/bin/env python
# -*- coding:utf-8 -*-
import PIL.Image as Image

if __name__ == "__main__":
    dir = ""
    dstfn = dir+"mosaic_img.jpg"
    height = 20
    width = 2
    piece_size = 256
    dstImg = Image.new("RGBA", (piece_size*width, piece_size*height))

    for y in range(height):
        for x in range(width):
            fn = dir+str(x)+"_"+str(y)+".jpg"
            img = Image.open(fn)
            dstImg.paste(img, (x*piece_size, y*piece_size))
        print ("finished row-%d"%(y+1))

    dstImg.save(dstfn)