import pandas as pd


class PartyLabeler:
    def __init__(self):
        # Dictionary of president names to their political party affiliations
        self.president_to_party = {
            'Lyndon B. Johnson': 'Democrat',
            'Ronald Reagan': 'Republican',
            'Barack Obama': 'Democrat',
            'Franklin D. Roosevelt': 'Democrat',
            'John F. Kennedy': 'Democrat',
            'Donald Trump': 'Republican',
            'George W. Bush': 'Republican',
            'Bill Clinton': 'Democrat',
            'Woodrow Wilson': 'Democrat',
            'Ulysses S. Grant': 'Republican',
            'Andrew Johnson': 'National Union',
            'Herbert Hoover': 'Republican',
            'Grover Cleveland': 'Democrat',
            'Andrew Jackson': 'Democrat',
            'James K. Polk': 'Democrat',
            'Thomas Jefferson': 'Democratic-Republican',
            'Richard M. Nixon': 'Republican',
            'George H. W. Bush': 'Republican',
            'Benjamin Harrison': 'Republican',
            'James Madison': 'Democratic-Republican',
            'Jimmy Carter': 'Democrat',
            'Theodore Roosevelt': 'Republican',
            'George Washington': None,  # No party affiliation
            'Joe Biden': 'Democrat',
            'Harry S. Truman': 'Democrat',
            'Warren G. Harding': 'Republican',
            'John Tyler': 'Whig',
            'Rutherford B. Hayes': 'Republican',
            'Franklin Pierce': 'Democrat',
            'Abraham Lincoln': 'Republican',
            'Gerald Ford': 'Republican',
            'William McKinley': 'Republican',
            'James Buchanan': 'Democrat',
            'William Taft': 'Republican',
            'Calvin Coolidge': 'Republican',
            'Chester A. Arthur': 'Republican',
            'Martin Van Buren': 'Democrat',
            'James Monroe': 'Democratic-Republican',
            'John Adams': 'Federalist',
            'John Quincy Adams': 'Democratic-Republican',
            'Millard Fillmore': 'Whig',
            'Dwight D. Eisenhower': 'Republican',
            'Zachary Taylor': 'Whig',
            'William Harrison': 'Whig',
            'James A. Garfield': 'Republican'
        }

    def add_party_affiliation(self, df):
        # Create a new column 'party' and initialize with None
        df['party'] = None

        # Update the 'party' column for each president based on the dictionary
        for president, party in self.president_to_party.items():
            df.loc[df['name'] == president, 'party'] = party

        return df