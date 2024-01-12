# -*- coding: utf-8 -*-

import pandas as pd
import re

class TranscriptPreprocessor:
    def __init__(self):
        # Define replacement patterns
        self.replacement_patterns = [
            # Remove mentions of "View Transcript" and similar
            ('View Transcript', ' ', False),
            ('Transcript', ' ', False),
            
            # Replace newline and non-breaking space characters with spaces
            ('\n', ' ', True),
            ('\xa0', ' ', True),
            
            # Remove formal introductions and mentions of various presidents
            ('By the President of the United States of America', ' ', False),
            ('Proclamation', ' ', False),
            ('THE PRESIDENT:', ' ', False),
            ('President Clinton:', ' ', False),
            ('PRESIDENT TRUMP', ' ', False),
            ('THE PRESIDENT.', ' ', True),
            ('THE PRESIDENT said:', ' ', False),
            ('President Bush.', ' ', True),
            
            # Remove audience reactions
            ('Audience: No-o-o!', ' ', False),
            ('Audience members. Boo-o-o!', ' ', True),
            ('Audience members. Viva Bush! Viva Bush! Viva Bush!', ' ', True),
            ('Audience members. U.S.A.! U.S.A.! U.S.A.!', ' ', True),
            ('AUDIENCE: H.R.3! H.R.3! H.R.3!', ' ', True),
            
            # Remove presidential statements
            ('THE PRESIDENT made the following statement:', ' ', False),
            ('IN REPLY to press questions as to the business situation the President said:', ' ', False),
            
            # Remove applause and laughter notations
            ('\[applause\]', '', True),
            ('\[Laughter\]', '', True),
            ('\[Applause\]', '', True),
            ('\(Applause.\)', '', True),
            ('\(Laughter.\)', '', True),
            ('\(Audience interruption.\)', '', True),
            ('\(APPLAUSE\)', '', True),
            ('\(applause\)', '', True),
            
            # Replace curly apostrophes with straight ones
            ('’', '\'', True),
            
            # Replace double spaces with a single space
            ('  ', ' ', True)
        ]
        
    def replace_patterns(self, text):
        for pattern, replacement, is_regex in self.replacement_patterns:
            if is_regex:
                text = re.sub(pattern, replacement, text)
            else:
                text = text.replace(pattern, replacement)
        return text
    
    def preprocess_transcripts(self, df):
        # Apply preprocessing to the 'transcript' column of the DataFrame
        df['transcript'] = df['transcript'].apply(self.replace_patterns)
        df['transcript'] = df['transcript'].str.strip()
        df['transcript'] = df['transcript'].str.lower()
        return df
    
# Usage Example
# preprocessor = TranscriptPreprocessor()
# preprocessed_df = preprocessor.preprocess_transcripts(original_df)