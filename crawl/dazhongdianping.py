# !/usr/bin/env python
# -*- coding:utf-8 -*-
from urllib import request, parse
import urllib, codecs, json

# url: http://www.dianping.com/search/map/category/4/10/g207o6#

class DZDP:

    def __init__(self, cityId='4', cityEnName = "guangzhou", regionId='0', category="207", shopSortItem = '6', fn = "test.csv"):
        self.cityId=cityId
        self.cityEnName = cityEnName
        self.regionId = regionId
        self.category = category
        self.shopSortItem = shopSortItem
        self.pageCount = 1
        self.baseurl = "http://www.dianping.com/search/map/ajax/json"

        self.fn = fn
        self.f = codecs.open(self.fn, 'w', 'utf-8_sig')
        self.f.write("city,district,date,address,avg_price,booking_setting,tag,lat,lng,phone_NO,shop_id,shop_name,shop_power\n")

        self.params = []
        self.params = [("cityId", self.cityId), ("cityEnName", self.cityEnName),
                       ("promoId", "0"), ("shopType","10"), ("categoryId", self.category),
                       ("regionId", self.regionId), ("sortMode", "2"), ("shopSortItem", self.shopSortItem),
                       ("keyword", ""), ("searchType", "1"), ("branchGroupId", "0"), ("shippingTypeFilterValue", "0"),("page", "1")]

    def get_html(self, page):
        data = self.params
        data[-1] = ("page", str(page))
        postdata = parse.urlencode(data)
        print (postdata)
        req = request.Request(self.baseurl)

        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; WOW64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/52.0.2743.116 Safari/537.36")
        req.add_header("Origin", "http://www.dianping.com")
        try:
            resp = request.urlopen(req, data=postdata.encode("utf-8"))
            resp = resp.read().decode("utf-8")
            resp = resp.replace("\\","")

            results = json.loads(resp)

            self.pageCount = int(results["pageCount"])
            if self.pageCount > 50:
                self.pageCount = 50
            return results["shopRecordBeanList"]
        except:
            return


    def parse_results(self, results):
        print (results)
        for item in results:
            data = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(self.cityEnName,item["shopRecordBean"]["districtName"],item["addDate"],item["address"],item["avgPrice"],
                                        item["bookingSetting"],item["dishTag"],item["geoLat"],item["geoLng"],item["phoneNo"],
                                        item["shopId"],item["shopName"],item["shopPower"])
            self.f.write(data)
            self.f.flush()


    def main(self):
        rs = self.get_html(1)
        dp.parse_results(rs)
        if self.pageCount > 1:
            for i in range(2, self.pageCount+1):
                rs = self.get_html(i)
                dp.parse_results(rs)
        self.f.close()

if __name__ == "__main__":
    dp = DZDP(cityEnName="广州")
    dp.main()

