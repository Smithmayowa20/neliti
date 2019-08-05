from sickle import Sickle
from langdetect import detect
import json
import os
import requests
import random
import uuid
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import pandas as pd
import click
	

class Neliti:
    """
    A class that instantiates objects that access and store research 
	entries from the neliti oai-pmh api database, and further calculate
	the word movement distance of the words of each entries description 
	to determine the most similar entries for each entry.
    """
	
	
    def __init__(self, subscription_key=''):
        self.database = []
        self.url = 'https://www.neliti.com/oai'
        self.subscription_key = subscription_key
        self.output = {}
        self.vectors = []
        self.model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', limit=50000) # Inits the model
        self.stopWords = set(stopwords.words('english'))

    
    def _get_translation(self, text_input, language_output="en"):
        """
        Given a record's description in indonesian this function translates 
	    such description from indonesian to english with azure translation api
        """
	
        base_url = 'https://api.cognitive.microsofttranslator.com'
        path = '/translate?api-version=3.0'
        params = '&to=' + language_output
        constructed_url = base_url + path + params

        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        # You can pass more than one object in body.
        body = [{
            'text' : text_input
        }]
        response = requests.post(constructed_url, headers=headers, json=body)
        return response.json()
	

    def _get_record_data(self, record):
        """
        Given a single neliti entry record instance, this function parses and 
	    returns the entry metadata details.
        """
	
        try:
            description = (record.metadata)['description']
            description = self._detect_description(description[0])
            if not(description):
                return False
            url = (record.metadata)['identifier']
            type = (record.metadata)['type']
            date = (record.metadata)['date']
            title = (record.metadata)['title']
            publisher = (record.metadata)['publisher']
            creator = (record.metadata)['creator']
            subject = (record.metadata)['subject']
            source = (record.metadata)['source']
        except KeyError:
            return False
				
        data = {
			'url' : url,
			'type' : type,
			'date' : date,
			'title' : title,
			'publisher' : publisher,
			'description' : description,
			'creator' : creator,
			'subject' : subject,
			'source' : source
	    }
        return(data)
		
		
		
    def _get_database(self, number):
        """
        This object method makes an api call to neliti api and iteratively 
	    yields each record entry for processing, keeping track of the total no 
	    of records that has been successfully processed.
        """
	
        sickle = Sickle(self.url)
        records = sickle.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
        percentage = 0
        no_of_records = 0
        while 1:
            record = records.next()
            if(record):
                pass
            else:
                break
            data = self._get_record_data(record)
            if data:
                self.database.append(data)
                no_of_records += 1
            else:
                continue
            if ((no_of_records % 100 == 0) and (no_of_records != 0)):
                print("Progress : {no_of_records} records Downloaded".format(no_of_records=no_of_records))
            if (no_of_records == number):
                break
			
				
				
    def _detect_description(self, description):
        """
        Given an entry's description it detects the description
	    language and translates it to english if it is indonesian,
	    It also fixes the error where indonesian and english text
	    are in the same entry's description.
        """
	
        if (('.English' in description) or ('.Indonesian' in description)):
            return False
            print(description)
            description = input(">>>\n")
            print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(description)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            return description
		
        if (detect(description) == "id"):
            translate = self._get_translation(description)
            try:
                eng_description = translate[0]['translations'][0]['text']
                if eng_description:
                    return(eng_description)
            except KeyError:
                return False
			
        elif (detect(description) == "en"):
            return description
			
			
    def _process(self, description):
        """
        Given an entry's description it returns a list of the
	    description words filtering out all the stop words.
        """
	    
        sentence_list = [word for word in description if not(word in self.stopWords)]
        return sentence_list


    def _mean_vector(self, sentence_list):
        """
        Given a list of an entry's description text words it 
	    returns the mean vector of all the words in the list.
        """
	
        vector = np.zeros(300)
        no_null = 0
        for i in range(len(sentence_list)):
            try:
                vector += self.model[sentence_list[i]]
            except KeyError:
                no_null += 1
        return(vector/(len(sentence_list) - no_null))


    def _cosine_similarity(self, desc_1, desc_2):
        mean_vector_1 = mean_vector(_process(desc_1))
        mean_vector_2 - mean_vector(_process(desc_2))
    
        return(cosine_similarity(mean_vector_1,mean_vector_2))


    def _wmd_distance(self, desc_1, desc_2):
        """
        Given an two entries description text it returns the
	    value of the word mover distance between both texts.
        """
	
        return(self.model.wmdistance(self._process(desc_1), self._process(desc_2)))


    def _create_batch_set(self, batch):
        """
        It returns the unique set of possible entries description 
	    comparism between all the entries description in a batch
	    """
	
        return({frozenset((i,j)) for i in range(len(batch) - 1) for j in range(len(batch)) if i != j})


    def _calculate_batch_wmd(self, batch, batch_set):
        """
        It calculates and returns the word mover distance value
	    between two entries description.
        """
        for entry in batch_set:
            values = []
            for j in entry:
                values.append(j)
            wmd = self._wmd_distance(batch[values[0]]['description'], batch[values[1]]['description'])
		
		
            query_1 = (batch[values[0]]['title'][0])
            db_value_1 = self.output.get(query_1, {})
            if db_value_1:
                self.output[query_1][batch[values[1]]['title'][0]] = wmd

            else:
                self.output[query_1] = {}
                self.output[query_1][batch[values[1]]['title'][0]] = wmd
		
		
            query_2 = (batch[values[1]]['title'][0])	
            db_value_2 = self.output.get(query_2, {})
            if db_value_2:
                self.output[query_2][batch[values[0]]['title'][0]] = wmd

            else:
                self.output[query_2] = {}
                self.output[query_2][batch[values[0]]['title'][0]] = wmd
        

    
    def _batch_db(self, lis, batch_size):
        """
        Provided a randomized database list and a batch_size
	    which is also a factor of the total length of the database
	    list, it returns a list of the batches according to the 
	    provided batch size.
        """
	
        return ([lis[x:x+batch_size] for x in range(0,len(lis),batch_size)])

	
    def _batch_db_1(self, lis, batch_size):
        """
        Provided a randomized database list and a batch_size
	    which is not a factor of the total length of the database
	    list, it returns a list of the batches according to the 
	    provided batch size.
        """
	
        prefix_lis = [lis[x:x+batch_size] for x in range(0,len(lis),batch_size)]
        suffix_lis = lis[((len(lis)//batch_size) * batch_size):len(lis)]
        prefix_lis.append(suffix_lis)
        return(prefix_lis) 
	
	
    def _get_db_mean_values(self):
        """
        Calculates and stores the mean vector of all entries
		description in the neliti stored database.
        """
        for entry in self.database:
            description = entry['description']
            mean_vector = self._mean_vector(self._process(description))
            self.vectors.append((entry['title'][0], mean_vector.tolist()))


    def _get_batch_wmd_distance(self, batch_size):
        """
        It randomizes the database list and also calculates the word
	    mover distance for all the database entries in relation to
	    their batch size.
        """
	
        lis = list(range(len(self.database)))
        self._get_db_mean_values()
        random.shuffle(lis)
        lis = self._batch_db(lis, batch_size) if (len(lis)%batch_size == 0) else self._batch_db_1(lis, batch_size)	
        for batch in lis:
            batch = [self.database[i] for i in batch]
            batch_set = self._create_batch_set(batch)
            self._calculate_batch_wmd(batch,batch_set)
		
		
    def _save_output(self):
        """
        It serializes and saves the word mover distance output dict into
	    a json file.
        """
	
        with open('output.json', 'w') as F:
            F.write(json.dumps(self.output, indent=4))	
		
    def _save_vectors(self):
        """
        It serializes and saves the vector list into a json file.
        """
        file_path = 'vectors.json'
        pd.Series(self.vectors).to_json(file_path, orient='values')		
		
    def _save_database(self):
        """
        It serializes and saves the entries database list into a json file.
        """
	
        with open('database.json', 'w') as F:
            F.write(json.dumps(self.database, indent=4))
			
		
    def start_process(self, number, batch_size):
        """
        Provided the number of neliti database entries required and
	    the batch_size it starts the program execution.
        """
	
        self._get_database(number)
        self._get_batch_wmd_distance(batch_size)
        self._save_output()
        self._save_database()
        self._save_vectors()
      

@click.command()
@click.option('--number', default=1000, help='Number of neliti database entries required')
@click.option('--batch_size', default=10, help='Batch size for calculating wmd')	
def execute(number, batch_size):
    neliti = Neliti()
    neliti.start_process(number,batch_size)
	
if __name__ == '__main__':
    execute()