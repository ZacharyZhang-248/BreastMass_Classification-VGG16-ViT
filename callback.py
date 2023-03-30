class My_ASK(keras.callbacks.Callback):
    def __init__(self, model, epochs, ask_epoch, dwell=True, factor=.4):
        super(My_ASK, self).__init__()
        self.model = model
        
        """
	After ask_epoch training, we can choose whether to halt or continuing training. 
        If continues, an integer needs to be typed indicating following training epochs.
	"""
        
        self.ask_epoch = ask_epoch
        
        self.epochs = epochs
        self.ask = True # whether to ask
        self.lowest_vloss = np.inf
        self.lowest_loss = np.inf
        self.best_weights = self.model.get_weights() # optimial weights is initialized to initial model weights
        self.best_epoch = 1
        self.vlist = [] # list storing change of validation loss
        self.tlist = [] # list storing change of training loss
        self.dwell = dwell
        self.factor = factor # decay factor of learning factor
        
        
    def on_train_begin(self, logs = None):
        if self.ask_epoch == 0:
            print('You set ask_epoch = 0, ask_epoch will be set to 1', flush = True)
            self.ask_epoch = 1
        if self.ask_epoch >= self.epochs: 
            print('ask_epoch >= epochs, will train for ', epochs, ' epochs', flush=True)
            self.ask = False
        if self.epochs == 1:
            self.ask = False
        else:
             
            print(f'Training will proceed until epoch {ask_epoch} then you will be asked to')
            print('Enter H to halt training or enter an integer for how many more epochs to run then be asked again')
            
            if self.dwell:
                print('\n Learning rate will be automatically adjusted during training')
                
        self.start_time = time.time()
        
        
    def on_train_end(self, logs=None):
        print(f'Loading model with weights from epoch {self.best_epoch}')
        
        self.model.set_weights(self.best_weights)
        train_duration = time.time() - self.start_time
        hours = train_duration // 3600
        minutes = (train_duration - hours * 3600) // 60
        seconds = train_duration - hours * 3600 - minutes * 60

        print(f'Training using {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds')

        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        loss = logs.get('loss')
        if epoch > 0:
            delta_v = self.lowest_vloss - val_loss # D-value between current validation loss and lowest validation loss
            vimprov = (delta_v / self.lowest_vloss) * 100 # percentage of improvement, loss increased if this value is negative
            self.vlist.append(vimprov)
            
            delta_t = self.lowest_loss - loss
            timprov = (delta_t / self.lowest_loss) * 100
            self.tlist.append(timprov)
        else:
            vimprov = 0.0
            timprov = 0.0
        
        if val_loss < self.lowest_vloss:
            self.lowest_vloss = val_loss # update lowest validation loss
            self.best_weights = self.model.get_weights() # update optimal model weights
            self.best_epoch = epoch + 1
            print(f'\n Validation loss of {val_loss:7.4f} is {vimprov:7.4f} % below lowest loss, saving weights from epoch {str(epoch + 1):3s} as best weights')
        else:
            vimprov = abs(vimprov)
            print(f'\n Validation loss of {val_loss:7.4f} is {vimprov:7.4f} % above lowest loss of {self.lowest_vloss:7.4f}. Keeping weights from epoch {str(self.best_epoch)} as best weights')
            
            if self.dwell:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = lr * self.factor
                print(f'\n Learning rate was automatically adjusted from {lr:8.6f} to {new_lr:8.6f}, model weights set to best weights')
                
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.model.set_weights(self.best_weights)
        
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            
        if self.ask:
            if epoch + 1 == self.ask_epoch:
                print('\n Enter H to end training or an integer for the number of additional epochs to run then ask again')
                ans = input()
                
                if ans == 'H' or ans == 'h' or ans == '0': # stop training
                    self.model.stop_training = True
                else:
                    self.ask_epoch += int(ans) # ask at ask_epoch+ans epoch again
                    if self.ask_epoch > self.epochs:
                        print('\n Your specification exceeds ', self.epochs, ' cannot train for ', self.ask_epoch, flush =True)
                    else:
                        print(f'\n You entered {ans}. Training will continue to epoch {self.ask_epoch}')
                        
                        if self.dwell == False:
                            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) 
                            print(f'\n Current LR is  {lr:8.6f}  hit enter to keep  this LR or enter a new LR')
                            
                            ans = input(' ')
                            if ans == '':
                                print(f'\n Keeping current LR of {lr:7.5f}')
                                
                            else:
                                new_lr = float(ans)
                                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                                print(f'\n Changing LR to {ans}')
