import scrapy
from ..items import MessagespiderItem


class MesSpider(scrapy.Spider):
    name = "mes"
    allowed_domains = ["itcast.cn"]
    start_urls = ["http://www.itcast.cn/"]

    def parse(self, response):
        items = []
        for item in response.xpath("//div[@clas='li_txt']"):
            i = MessagespiderItem()
            name = item.xpath("h3/text()").extract()
            title = item.xpath("h4/text()").extract()
            info = item.xpath("h5/text()").extract()
            i["name"] = name
            i["title"] = title
            i["info"] = info
            print(i)
            pass

