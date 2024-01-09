# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time


class ContentScraper:
    def scrape_speech_content(self, url):
        # Start a new Chrome driver session
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        # Navigate to the speech URL
        driver.get(url)

        # Wait for the webpage to fully load
        driver.implicitly_wait(2)

        # Maximize the browser window (optional)
        driver.maximize_window()

        # Extract the page source
        html = driver.page_source

        # Close the browser session
        driver.quit()

        # Parse the webpage content
        soup = BeautifulSoup(html, 'html.parser')

        # Extract details
        name = soup.select_one('p.president-name').text.strip()
        date = soup.select_one('p.episode-date').text.strip()
        title = soup.select_one('h2.presidential-speeches--title').text.strip()
        transcript = soup.select_one('div.view-transcript').text.strip()
        
        return pd.DataFrame({'name': name, 'date': date, 'title': title, 'transcript': transcript}, index=[0])


# Example usage
# scraper = SpeechContentScraper()
# data = scraper.scrape_speech_content('some_url')
# print(data)