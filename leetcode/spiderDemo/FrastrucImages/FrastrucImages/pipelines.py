# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy import Request


class FrastrucimagesPipeline:
    def get_media_requests(self, item, info):
        file_meta = {}
        file_meta['url_name'] = item['url_name']
        file_meta['url_seq'] = item['url_seq']
        yield Request(item['url'], meta=file_meta)

    def file_path(self, request, response=None, info=None):
        url = request.url
        format = url.split('.')[-1]
        file_meta = request.meta
        file_name = file_meta['url_name'] + '_' + str(file_meta['url_seq']) + '.' + format
        return file_name
