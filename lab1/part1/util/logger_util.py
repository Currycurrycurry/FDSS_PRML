import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s',
                    filename='./logs/{}.log'.format(datetime.today()), filemode='w')  # %(asctime)s - %(name)s -
logger = logging.getLogger(__name__)
