# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Liu Penghua
# date: 2017-7-14

# 功能： 抓取百度poi
# 输入参数：城市，关键字，密钥，输出文件名
# 未认证日配额只有2000次，可申请多个密钥
from urllib import request
import urllib, codecs, math

class BDPOI:

    def __init__(self, city, keywords, apikey, outfn):
        self.city = city
        self.keywords = keywords
        self.apikey = apikey
        self.outfn = outfn
        self.baseurl = "http://api.map.baidu.com/place/v2/search?"
        self.f = codecs.open(self.outfn,'w', 'utf-8_sig')
        self.f.write("uid,name,lat,lng,address,type,tag\n")
        self.total = 0
        self.maxpage = 1

    def set_params(self, coord_type = 1, page_size = 20):
        self.coord_type = coord_type
        self.page_size = page_size

    def get_html(self, num=0):
        query = {
            'ak': self.apikey,
            'query': self.keywords,
            'region':self.city,
            'coord_type': self.coord_type,
            'page_size': str(self.page_size),
            'page_num': str(num),
            'output': 'json',
            'scope': '2'
        }

        param = urllib.parse.urlencode(query)
        _url = self.baseurl + param
        print(_url)

        req = request.Request(_url)
        req.add_header(key="User-Agent", val="Mozilla/5.0 (Windows NT 10.0; WOW64) "
                                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                                         "Chrome/52.0.2743.116 Safari/537.36")


        res = request.urlopen(req)
        if res.reason == "OK":
            html = res.read().decode("utf-8")
            # print (html)
            html = (eval(html))
            self.total = html["total"]
            self.maxpage = math.ceil(int(self.total)/20)
            results =  html["results"]
            print (results)
            return results
        else:
            print (res.reason)
            return

    def parser(self, results):
        for poi in results:
            loc=poi['location']
            poi['lat']=loc['lat']
            poi['lng']=loc['lng']
            poi['type']=''
            poi['tag']=''
            if 'detail_info' in poi.keys():
                details=poi["detail_info"]
                if 'type' in details.keys():
                    poi['type']=details['type']
                if 'tag' in details.keys():
                    poi['tag']=details['tag']

            _data = '%s,%s,%s,%s,%s,%s,%s\n' % (
                poi['uid'], poi['name'], poi['lat'], poi['lng'], poi['address'], poi['type'], poi['tag'])
            #print (_data)
            self.f.write(_data)
            self.f.flush()

    def main(self):
        self.set_params(coord_type=1, page_size=20)
        rs = self.get_html(num=0)
        self.parser(rs)
        if (self.maxpage > 1):
            for i in range(1, self.maxpage):
                rs = self.get_html(num=i)
                if len(rs)==0:
                    break
                self.parser(rs)
        self.f.close()


if __name__ == "__main__":
    baidupoi = BDPOI(u"广州",u"大学","your_api_key","test.csv")
    baidupoi.main()
