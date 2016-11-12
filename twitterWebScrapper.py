'''
This code tries to webscrap a special hashtag from the twitter and extract all the User_id's who tweeted that specific hashtag
and will save them all in a text file

Using this code is allowed just with referencing the authur

Mojtaba Sedigh Fazli, OCT 2016
'''

import urllib2
import re
from bs4 import BeautifulSoup
#import time for tracing the error
import time
import argparse
import datetime
import os
from operator import itemgetter

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import sys

import unittest, time, re

class Sel(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.base_url = "https://twitter.com"
        self.verificationErrors = []
        self.accept_next_alert = True
    def test_sel(self):
        driver = self.driver
        delay = 3
        driver.get(self.base_url + "/search?q=%23VitalSigns&src=typd")
        #driver.find_element_by_link_text("All").click()
        userids=set()
        for i in range(1,100):

            html_source = driver.page_source
            the_page = html_source.encode('utf-8')
            soup = BeautifulSoup(the_page, 'html.parser')


            d = soup.find_all('div', attrs={'class':'tweet js-stream-tweet js-actionable-tweet js-profile-popup-actionable original-tweet js-original-tweet '}) 
            for t in d:
                userids.add(t['data-user-id',''])
                t.find('a',attrs={'class':''})
            with open('current_userids.txt','w') as ff:
                for userid in userids:
                    ff.write('%s\n' % str(userid))

            print(len(userids))
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)



if __name__ == "__main__":
    unittest.main()
