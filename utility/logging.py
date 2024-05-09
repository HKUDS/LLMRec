import os
from datetime import datetime
# Logger class for logging
class Logger():
    # Constructor
    def __init__(self, filename, is_debug, path='./logs/'):
        # Set the filename
        self.filename = filename
        self.path = path
        self.log_ = not is_debug
    # Function to log the input
    def logging(self, s):
        # Convert the input to a string
        s = str(s)
        # Print the input
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        # If logging is enabled
        if self.log_:
            # Write the input to the log file
            with open(os.path.join(os.path.join(self.path, self.filename)), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')



