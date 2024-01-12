# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time


class URLScraper:
    def __init__(self, url):
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.url = url
        self.speech_urls = []
        
    def load_page(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(2)
        self.driver.maximize_window()

    def scroll_page(self, scroll_count=50):
        for _ in range(scroll_count):
            self.driver.find_element(By.TAG_NAME,'body').send_keys(Keys.END)
            time.sleep(2)

    def extract_speech_urls(self):
        html = self.driver.page_source
        soup = BeautifulSoup(html,'html.parser')
        div = soup.select('div.views-row')
        for i in div:
            self.speech_urls.append(i.select_one('div.views-field.views-field-title > span.field-content > a')['href'])

    def close_driver(self):
        self.driver.quit()
