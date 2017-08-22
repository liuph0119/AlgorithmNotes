# !/usr/bin/env python
# -*- coding:utf-8 -*-
from urllib.request import Request, urlopen
from urllib import request
from bs4 import BeautifulSoup as bs
import re, sys, codecs, time, os
import gzip




def restart_program(param):
    _command = "E:\professional\Python36_32\python.exe F:\Code_Python\Projects\shenzhen_housePrice\\bj_soufun\ershoufang_bj.py"
    _command = _command + ' '+ str(param)
    os.system(_command)

def getHtml(_url):
    req = request.Request(_url)
    req.add_header(key="User-Agent", val="Mozilla/5.0 (Windows NT 10.0; WOW64) "
                                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                                         "Chrome/52.0.2743.116 Safari/537.36")
    resp = request.urlopen(req)
    # print (resp.status, resp.reason)
    if resp.reason=='OK':
        html = (resp.read())
        return html
    else:
        print (resp.reason)

def getXY(html, _pat_x, _pat_y):
    # 根据正则表达匹配查找经纬度
    x = re.findall(_pat_x, html)
    y = re.findall(_pat_y, html)
    lng = ""
    lat = ""
    # 如果匹配到了，就提取出来。因为查找到的是一个列表，所以需要提取其中的元素
    if len(x)>0:
        lng = x[0]
    if len(y)>0:
        lat = y[0]
    return lng, lat

def getComarea(_url):
    html = getHtml(_url)

    try:
        html = gzip.decompress(html).decode('utf-8')
    except:
        html = html.decode("utf-8")

    #print(eval(html))
    data = eval(html)
    comarea = {}
    for district in data:
        qu = district["name"]
        comarea[qu] = []
        subcomarea = []
        areas = (district["area"])
        for area in areas:
            # print (area["id"],area["name"])
            subcomarea.append(area["id"])
        comarea[qu] = (subcomarea)
    return comarea

def getlnglat(_newcode):
    reg_x = r'px:"(.+?)"'
    reg_y = r'py:"(.+?)"'
    pat_x = re.compile(reg_x)
    pat_y = re.compile(reg_y)

    mapurl = "http://esf.fang.com//newsecond/map/NewMapDetail.aspx?newcode=" + _newcode
    maphtml = getHtml(mapurl)
    try:
        maphtml = gzip.decompress(maphtml).decode('gbk')
    except:
        maphtml = maphtml.decode("gbk")
    lng, lat = getXY(maphtml, pat_x, pat_y)
    return lng,lat

def getMaxPage(html):
    soup = bs(html, "html.parser")
    try:
        # 首先获取最大页数
        pages = soup.find("div", "fanye").findAll("a")
        maxpage = (pages[-1])["data-id"]
        maxpage = int(maxpage)
        # print("total page: ", maxpage)
    except:
        maxpage = 1
    return maxpage

def parse_extract(html):
    soup = bs(html, "html.parser")

    houses = soup.find("div", "houseList").findAll("dl", "list rel")
    #print("items: ", len(houses))
    for house in houses:
        # 查找每一个房源的链接和经纬度（根据newcode）
        href = house.a["href"]
        newcode = house.a["data_id"]
        _lng, _lat = getlnglat(newcode)
        try:
            html = getHtml(href)
        except:
            print("href invaid: ",href)

        try:
            html = gzip.decompress(html).decode('gbk')
        except:
            html = html.decode("gbk")
        subsoup = bs(html, "html.parser")
        dls = subsoup.find("div", "inforTxt").findAll("dl")
        if len(dls)<2:
            continue

        price = dls[0].find("dt", "gray6 zongjia1")
        name = (house.a.dd.find("p", "title")).text.replace(",","，")
        try:
            zongjia = price.contents[3].text + price.contents[4].text
        except:
            zongjia = ''
        try:
            danjia = (price.contents[5].strip())[1:-1]
        except:
            danjia = ''
        dds = dls[0].findAll("dd", "gray6")
        num = 1

        try:
            huxing = dds[num].contents[1]
            num += 1
        except:
            huxing = ""

        try:
            jianzhumianji = dds[num].contents[1].text
            num += 1
        except:
            jianzhumianji = ""

        dds = dls[1].findAll("dd")

        info = {}
        info["年代："] = ""
        info["朝向："] = ""
        info["楼层："] = ""
        info["结构："] = ""
        info["装修："] = ""
        info["住宅类别："] = ""
        info["建筑类别："] = ""
        info["产权性质："] = ""
        if len(dds)<1:
            continue

        for dd in dds:
            if len(dd.contents)<2:
                continue
            try:
                info[dd.contents[1].text] = (dd.contents[2]).strip()
            except:
                pass
            # niandai = dds[num].contents[2]
            # chaoxiang = dds[num].contents[2]
            # louceng = dds[num].contents[2]
            # jiegou = dds[num].contents[2]
            # zhuangxiu = dds[num].contents[2]
            # zhuzhaileibie = dds[num].contents[2]
            # jianzhuleibie = dds[num].contents[2]
            # chanquanxingzhi = dds[num].contents[2]

        try:
            loupanmingcheng = (dls[1].find("dt").findAll("a"))[0].text.strip()
        except:
            loupanmingcheng = ""
        #print(huxing, jianzhumianji, info["年代："], info["朝向："], info["楼层："], info["结构："], info["装修："], info["住宅类别："], info["建筑类别："],
        #     info["产权性质："], loupanmingcheng)
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
         qu, name, zongjia, danjia, huxing, jianzhumianji, info["年代："], info["朝向："], info["楼层："], info["结构："], info["装修："], info["住宅类别："], info["建筑类别："],
              info["产权性质："], loupanmingcheng, _lng, _lat, href))
        f.flush()
        #time.sleep(0.5)




if __name__ == "__main__":
    params = sys.argv
    null = ''
    true = 'true'
    false = 'false'

    NEWCODE = '44' \
              ''
    fn = "ershoufang_beijing_朝阳" + NEWCODE + ".csv"
    f = codecs.open(fn, 'a', 'utf-8_sig')
    # f.write('区,房源名称,总价,单价,户型,建筑面积,年代,朝向,楼层,结构,装修,住宅类别,建筑类别,产权性质,楼盘名称,经度,纬度,链接\n')

    # 以下url查询北京市的片区, 上海的为"http://esf.sh.fang.com/map/"
    url0 = "http://esf.fang.com/map/?a=getDistArea&city=bj"
    vComarea = getComarea(url0)
    #print ("Comarea: ",vComarea)

    for qu in vComarea.keys():
        districts = vComarea[qu]
        print(qu, districts)
        #for comarea in districts:
        comarea = NEWCODE
        if comarea == NEWCODE:
        # 以下url查询某个片区的小区, 1121
        #url1 = "http://esf.fang.com/map/?mapmode=y&district=1&comarea=1121&orderby=30&a=ajaxSearch&city=bj&searchtype=loupan"
            url1 =  "http://esf.fang.com/map/?mapmode=y&comarea="+str(comarea)+"PageNo=1&orderby=30&a=ajaxSearch&city=bj&searchtype=loupan"
            html = getHtml(url1)
            try:
                html = gzip.decompress(html).decode('gbk')
            except:
                html = html.decode("gbk")
            html = html.replace("\\/", "/")
            data = (eval(html))
            data = "<html><head>"+data["list"]+"</head><body></body></html>"
            print (data)
            max_page = getMaxPage(data)
            print ("max page: ",max_page)

            if (max_page >= 1):
                init_page = 1
                step_page = init_page
                if (len(params)>1):
                    init_page = int(params[1])
                try:
                    for i in range(init_page, max_page + 1):
                        step_page = i
                        url1 = "http://esf.fang.com/map/?mapmode=y&district=1&comarea=" + str(
                            comarea) + "&orderby=30&PageNo="+str(i)+"&a=ajaxSearch&city=bj&searchtype=loupan"
                        html = getHtml(url1)
                        try:
                            html = gzip.decompress(html).decode('gbk')
                        except:
                            html = html.decode("gbk")
                        html = html.replace("\\/", "/")

                        data = eval(html)
                        data_list = data['list']
                        data_list = "<html><head></head><body>" + data_list + "</body></html>"
                        sys.stdout.write("\rpage: %s/%s" % (i,max_page))
                        parse_extract(data_list)
                except:
                    param = str(step_page)
                    restart_program(param)
                print()
        sys.exit(0)
    f.close()


            #if(true):
                # 以下url查询房源, 1010026073
                #url2 = "http://esf.fang.com/map/?mapmode=y&orderby=30&newCode="+code+"&ecshop=ecshophouse&PageNo=1&a=ajaxSearch&city=bj&searchtype=fangyuan"
                # url = "http://esf.fang.com/map/?mapmode=y&district=&subwayline=&subwaystation=&price=&" \
                #      "room=&area=&towards=&floor=&hage=&equipment=&keyword=&comarea=2311&orderby=30&" \
                #       "isyouhui=&x1=116.050696&y1=39.801502&x2=116.740594&y2=40.058227&newCode=&houseNum=&" \
                #       "schoolDist=&schoolid=&ecshop=ecshophouse&PageNo=1&zoom=16&a=ajaxSearch&city=bj&searchtype=loupan"
                # html = getHtml(url2)
                # try:
                #     html = gzip.decompress(html).decode('gbk')
                # except:
                #     html = html.decode("gbk")
                # html = html.replace("\\/","/")
                #
                # data = eval(html)
                # data_list = data['list']
                # data_list = "<html><head></head><body>"+data_list+"</body></html>"
                #
                #
                # soup = bs(data_list, "html.parser")
                # try:
                #     pages = soup.find("div", "fanye").findAll("a")
                #     maxpage = (pages[-1])["data-id"]
                #     maxpage = int(maxpage)
                #     print ("total page: ", maxpage)
                # except:
                #     maxpage = 1
                #
                # parse_extract(data_list, lng, lat)
                #
                # if maxpage == 1:
                #     continue
                #
                # for i in range(2, maxpage+1):
                #     url2 = "http://esf.fang.com/map/?mapmode=y&orderby=30&newCode=" + code + "&ecshop=ecshophouse&PageNo="+str(i)+"&a=ajaxSearch&city=bj&searchtype=fangyuan"
                #     html = getHtml(url2)
                #     try:
                #         html = gzip.decompress(html).decode('gbk')
                #     except:
                #         html = html.decode("gbk")
                #     html = html.replace("\\/", "/")
                #
                #     data = eval(html)
                #     data_list = data['list']
                #     data_list = "<html><head></head><body>" + data_list + "</body></html>"
                #     print(data_list)
                #     parse_extract(data_list, lng, lat)





