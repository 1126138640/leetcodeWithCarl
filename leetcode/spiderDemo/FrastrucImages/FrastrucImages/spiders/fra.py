import scrapy
from scrapy import Request
from urllib.parse import quote
from ..items import FrastrucimagesItem
import re
import json


class FraSpider(scrapy.Spider):
    name = "fra"
    allowed_domains = ["cn.bing.com"]
    start_urls = ["https://cn.bing.com/images/search"]
    url_info = {'url_seq': 1}

    def parse(self, response):
        src_list = response.xpath('//div[@class="bd-home-content-album-item-pic"]/@style').extract()
        for src in src_list:
            index_start = src.find('(')
            index_end = src.find(')')
            item = FrastrucimagesItem()
            item['src'] = [src[index_start+1:index_end]]
            yield item