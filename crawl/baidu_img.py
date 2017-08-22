# !/usr/bin/env python
# -*- coding:utf-8 -*-

# 功能：下载256*256的影像
# 存在的问题：不一定所有的参数都能获取图像
# url: http://shangetu2.map.bdimg.com/it/u=x=779;y=156;z=12;v=009;type=sate&fm=46&udt=20150601
# url2:http://online0.map.bdimg.com/onlinelabel/?qt=tile&amp;x=12419&amp;y=3025&amp;z=16&amp;styles=pl&amp;udt=20170712&amp;scaler=1&amp;p=0
from urllib import request
import time


class BDIMG:

    # fn x=x-1542  y=318-y
    def __init__(self, x, y, z, fn):
        self.x = x
        self.y = y
        self.z = z
        self.fn = fn
        self.url = "http://shangetu1.map.bdimg.com/it/u=x=" + str(self.x) + ";y=" + \
                   str(self.y) + ";z=" + str(self.z) + ";v=009;type=sate&fm=46&udt=20170712"

    def get_html(self):
        req = request.Request(self.url)
        req.add_header(key="User-Agent", val="Mozilla/5.0 (Windows NT 10.0; WOW64) "
                                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                                             "Chrome/52.0.2743.116 Safari/537.36")

        res = request.urlopen(req)
        if res.reason == "OK":
            data = res.read()
            # print (data)
            f = open(self.fn, 'wb')
            f.write(data)
            f.close()
            print ("download img: %s"%self.fn)
        else:
            print (res.reason)
            return


if __name__ == "__main__":
    minx = 1532
    maxx = 1542
    miny = 298
    maxy = 318
    level = 13
    sleeptime = 2
    for x in range(minx, maxx+1):
        for y in range(miny, maxy+1):
            fn = str(x-minx) + "_"+ str(maxy-y)+".jpg"
            bdimg = BDIMG(x,y,level,fn)
            bdimg.get_html()
            time.sleep(sleeptime)

