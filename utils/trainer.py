import datetime
import numpy as np
import tensorflow as tf

class BaseTrainer():
    """
    class for handling the training procedure
    """
    
    def __init__(self, input_dict, output_dict, label_dict, losses_dict, train_gen, val_gen, 
                 steps_per_epoch_train, steps_per_epoch_val, config_file, out_scope: str='Encoding'):
        """
        initialize trainer
        
        args:
            input_dict: dictionary with graph input nodes
            output_dict: dictionary with graph output nodes
            label_dict: dictionary with graph label input nodes
            losses_dict: dictionary with losses. each key in dict need corresponding key in output_dict and label_dict
            train_gen: generator for training data
            val_gen: generator for validation data
            steps_per_epoch_train: int, number of batches per epoch
            steps_per_epoch_val: int, number of steps per loop over validation generator
            config_file: config file
            out_scope: str, name of scope of desired graph output
            
        returns:
            None
        """
        
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.label_dict = label_dict
        self.losses_dict = losses_dict
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_val = steps_per_epoch_val
        self.out_scope = out_scope
        
        self.params = config_file._sections['TRAINING']
    
    
    def train(self, checkpoint=None):
        """
        Perform training of model
        
        RESUME TRAINING FROM CHECKPOINT NOT YET IMPLEMENTED!!!!
        """
        
        # compute costs and define training optimization operation
        costs = [self.losses_dict[key](self.label_dict[key], self.output_dict[key], **self.params) 
                 for key in self.output_dict.keys()]
        cost = tf.reduce_sum(tf.concat(costs, axis=0))
        train_ops = tf.train.AdamOptimizer(float(self.params['lr'])).minimize(cost)
        
        # create summaries for tensorboard
        with tf.name_scope('summary'):
            train_summary = tf.placeholder(tf.float32, shape=None, name='train_loss_placeholder')
            train_loss_summary = tf.summary.scalar('train_loss', train_summary)
            val_summary = tf.placeholder(tf.float32,shape=None,name='val_loss_placeholder')
            val_loss_summary = tf.summary.scalar('val_loss', val_summary)
        loss_summaries = tf.summary.merge([train_loss_summary, val_loss_summary])
        
        # get name of output node
        out_node_name = [op.name for op in tf.get_default_graph().get_operations() 
                        if self.out_scope in op.name and not 'gradients' in op.name][-1]
        
        # initialize tf session and saver
        sess = tf.Session()
        saver = tf.train.Saver()
        
        # create run folder and initializer summary writer
        run_folder = 'runs/' + 'tf13' + str(datetime.datetime.now())
        writer = tf.summary.FileWriter(run_folder, sess.graph)
        
        # if a checkpoint has been parsed restore the session
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        
        # initialize variables
        sess.run(tf.global_variables_initializer())
        
        # start the training
        best_val_loss = np.inf
        patience_counter = 0
        for i in range(int(self.params['epochs'])):
            print('Epoch ' + str(i))
            
            train_loss = []
            for j in range(self.steps_per_epoch_train):
                
                # get batch and run optimization operation
                batch = self.train_gen.__next__()
                inp = self._build_input(batch)
                
                # if time for printing, evaluate cost and training optimization operation
                # else just evaluate training optimization operation
                if j % int(self.params['print_every']) == 0:
                    _, current_cost = sess.run([train_ops, cost], inp)
                    train_loss.append(current_cost)
                    print('Step ' + str(j) + ' | Training loss: ' + str(current_cost))
                else:
                    sess.run(train_ops, inp)
            
            # collect validation losses
            val_loss = []
            for j in range(self.steps_per_epoch_val):
                batch = self.val_gen.__next__()
                inp = self._build_input(batch)
                val_loss.append(sess.run(cost, inp))
            
            # average training and validation losses, write them to tensorboard and print 
            val_loss = np.mean(val_loss)
            train_loss = np.mean(train_loss)
            summ = sess.run(loss_summaries, feed_dict={train_summary: train_loss, val_summary: val_loss})
            writer.add_summary(summ, i+1)
            writer.flush()
            print('Epoch ' + str(i) + ' | Training loss: ' + str(train_loss) + ' | Validation loss: ' + str(val_loss))
            
            # update best validation loss if improvement and save checkpoint and model
            # if no improvement increment patience counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # save session
                saver.save(sess, run_folder + '/class_ckpt', i)
                output_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    tf.get_default_graph().as_graph_def(),
                    [out_node_name]
                    )
                
                # save graph
                with tf.gfile.GFile(run_folder + '/best_model.pb', "wb") as f:
                    f.write(output_graph_def.SerializeToString())
            else:
                patience_counter += 1
                if patience_counter >= int(self.params['patience']):
                    print('Training done!')
                    break
            
                    
    def _build_input(self, batch):
        """
        builds input for session evaluation from batch
        """
        
        inp = {}
        for key in self.input_dict.keys():
            inp[self.input_dict[key]] = batch[key]
        for key in self.label_dict.keys():
            inp[self.label_dict[key]] = batch[key]
        return inp
