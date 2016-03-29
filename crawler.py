import requests
import time
import os
import re
from multiprocessing import Process, Queue
from bs4 import BeautifulSoup

class LawCrawler:
	
	def __init__(self, category, num_process=1, verbose=False):
		self.num_process = num_process
		self.category = category
		self.verbose = verbose
		self.root_URL = "http://mobile.law.go.kr"
		if not os.path.exists("log/"):
			os.makedirs("log/")
		self.log = open("log/log_crawl.txt", "w")

	def __del__(self):
		self.log.close()

	def _get_law_list(self, category):
		"""
		Input
		- category: category string. It can be [lsSc, precSc, admRulsc, ...]

		Return
		- URL_list: URL list of each law
		"""
		URL_list = list()
		page_idx = 500
		base_req = self.root_URL+"/LSWM/mobile/"+category+".do?pageIndex="
		while (True):
			req = base_req+str(page_idx)
			ret = requests.get(req)
			soup = BeautifulSoup(ret.text, "html.parser")

			try:
				title_list = soup.find_all(lambda tag: tag.name == "span" and 
													   tag.get("class") == ["t1"])
				if not title_list: 
					raise
			
			except Exception as e:
				break
				
			URL_list.extend([t.a["href"] for t in title_list])

			if self.verbose and page_idx % 10 == 0:
				print("crawl {0} page" .format(page_idx))
			page_idx += 1

		self.log.write("{0} URLs are crawled\n" .format(len(URL_list)))

		return URL_list

	def _get_law(self, URL_list, output):
		"""
		Input
		- URL_list: url list of laws that made in self.get_law_list()
	
		Output (to queue)
		- data: string of entire laws. each laws are concatnated include <END> symbol.
				newline and tab are are replaced by " "
		"""   
		data = ""
		for url in URL_list:
			req = self.root_URL+url
			ret = requests.get(req)
			soup = BeautifulSoup(ret.text, "html.parser")

			try:
				law_title = soup.find(id="hgroup").h2.get_text()
				law_text = soup.find(lambda tag: tag.name == "div" and
											tag.get("class") == ["section"]).get_text()

			except Exception as e:
				self.log.write("Parse: {0}\n" .format(url))

			# law_text = re.sub('\s+',' ',law_text)
			data += law_title + "\n" + law_text + "<END>\n"
  
			if self.verbose is True:
			   print("{0}" .format(law_title))
			 

		output.put(data)

	def start(self):
		URL_list = self._get_law_list(self.category)

		work = int(len(URL_list) / self.num_process)
		todo = [ work*i for i in range(self.num_process) ]
		todo.append(len(URL_list)-1)

		process = []
		for i in range(self.num_process):
			q = Queue()
			p = Process(target=self._get_law, args=(URL_list[todo[i]:todo[i+1]], q))
			p.start()
			process.append([p, q])

		mp_data = []
		for i in range(self.num_process):
			mp_data.append(process[i][1].get())
			process[i][0].join()
			self.log.write("process {0} ended\n" .format(i))

		return "".join(mp_data)
