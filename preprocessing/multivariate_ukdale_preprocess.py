import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
# from functions import load_dataframe
import numpy as np
import os

DATA_DIRECTORY = 'UK_DALE/'
SAVE_PATH = 'created_data/UK_DALE/'
# Using Transformer project parameters (more accurate)
AGG_MEAN = 400
AGG_STD = 500

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning \
                                     example for NILM')
    parser.add_argument('--server', type=str, default='local', choices=['local', 'colab'],
                          help='Server environment: local or colab')
    parser.add_argument('--data_dir', type=str, default=None,
                          help='The directory containing the UKDALE data')
    parser.add_argument('--appliance_name', type=str, default='washingmachine',  # ---------------
                          help='which appliance you want to train: kettle,\
                          microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    args = parser.parse_args()
    
    # Set data_dir based on server environment if not explicitly provided
    if args.data_dir is None:
        if args.server == 'colab':
            args.data_dir = '/content/drive/MyDrive/dataset_dat/UK_DALE/'
        else:
            args.data_dir = DATA_DIRECTORY
    
    return args

# Using Transformer project parameters (based on actual UK-DALE data)
# Houses: 1, 2, 5 (same as Transformer project)
params_appliance = {
    'kettle': {
        'mean': 100,  # Transformer value (vs original 700)
        'std': 500,   # Transformer value (vs original 1000)
        'houses': [1],
        'channels': [10],  # House 1: ch10
    },
    'microwave': {
        'mean': 60,   # Transformer value (vs original 500)
        'std': 300,   # Transformer value (vs original 800)
        'houses': [1],
        'channels': [13],  # House 1: ch13
    },
    'fridge': {
        'mean': 50,   # Transformer value (almost perfect match to actual 47W!)
        'std': 50,    # Transformer value (perfect match to actual 50W!)
        'houses': [1],
        'channels': [12],  # House 1: ch12
    },
    'dishwasher': {
        'mean': 700,  # Transformer value (same as original)
        'std': 1000,  # Transformer value (same as original)
        'houses': [1],
        'channels': [6],  # House 1: ch6
    },
    'washingmachine': {
        'mean': 400,  # Transformer value (same as original)
        'std': 700,   # Transformer value (same as original)
        'houses': [1],
        'channels': [5],  # House 1: ch5
    }
}
def load_dataframe(directory, building, channel, col_names=['time', 'data'], nrows=None):
    df = pd.read_table(directory + 'house_' + str(building) + '/' + 'channel_' +
                       str(channel) + '.dat',
                       sep="\s+",
                       nrows=nrows,
                       usecols=[0, 1],
                       names=col_names,
                       dtype={'time': str},
                       )
    return df
args = get_arguments()
appliance_name = args.appliance_name
print(appliance_name)

def aggregate_app(df):
    a = df['aggregate']
    b = df[appliance_name]
    if a < b:
        b = a
    return b


def main():

    start_time = time.time()
    sample_seconds = 60
    training_building_percent = 0
    validation_percent = 20
    testing_percent = 20
    nrows = None
    debug = False  # Disabled plotting for faster execution

    # NILMFormer paper config: minute, hour, dow, month (4 features × 2 = 8 dimensions)
    train = pd.DataFrame(columns=['aggregate', appliance_name,
                                  'minute_sin', 'minute_cos',
                                  'hour_sin', 'hour_cos', 
                                  'dow_sin', 'dow_cos',
                                  'month_sin', 'month_cos'])

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + args.data_dir + 'house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')

        mains_df = load_dataframe(args.data_dir, h, 1)
        app_df = load_dataframe(args.data_dir,
                                h,
                                params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)],
                                col_names=['time', appliance_name]
                                )

        mains_df['time'] = pd.to_datetime(mains_df['time'], unit='s')
        mains_df.set_index('time', inplace=True)
        mains_df.columns = ['aggregate']
        #############################
        # mains_df.resample(str(sample_seconds) + 'S').mean()
        mains_df.reset_index(inplace=True)

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            print(mains_df.tail())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.savefig('original-mains.png')
            plt.show()

        # Appliance
        ############

        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')


        if debug:
            print("app_df:")
            print(app_df.head())
            print(app_df.tail())
            plt.plot(app_df['time'], app_df[appliance_name])
            plt.savefig('original-{}.png'.format(appliance_name))
            plt.show()

        # the timestamps of mains and appliance are not the same, we need to align them
        # 1. join the aggragte and appliance dataframes;
        # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)
        ######add 
        #app_df.resample(str(sample_seconds) + 'S').mean()


        df_align = mains_df.join(app_df, how='outer'). \
            resample(str(sample_seconds) + 'S').mean().bfill(limit=1)#bfill(limit=1)
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        
        # Extract temporal features from timestamp and apply sin/cos encoding
        # This preserves the cyclical nature of time (e.g., 23:00 is close to 00:00)
        # Based on NILMFormer README: "minutes, hours, days, months"
        # Note: Code uses hour/dow/month, but paper mentions minute
        minute = df_align['time'].dt.minute
        hour = df_align['time'].dt.hour
        dayofweek = df_align['time'].dt.dayofweek  # 0=Monday, 6=Sunday
        month = df_align['time'].dt.month
        
        # Sin/Cos encoding for cyclical features (matching NILMFormer approach)
        # 4 features × 2 (sin/cos) = 8 time dimensions
        df_align['minute_sin'] = np.sin(2 * np.pi * minute / 60.0)
        df_align['minute_cos'] = np.cos(2 * np.pi * minute / 60.0)
        df_align['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df_align['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        df_align['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
        df_align['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
        df_align['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df_align['month_cos'] = np.cos(2 * np.pi * month / 12.0)
        
        if appliance_name == 'fridge':
            df_align[appliance_name] = df_align.apply(aggregate_app,axis=1)

        # Select only needed columns (no timestamp, using sin/cos encoded time features)
        # NILMFormer paper: minute, hour, dow, month (8 time dimensions)
        df_align = df_align[['aggregate', appliance_name,
                             'minute_sin', 'minute_cos',
                             'hour_sin', 'hour_cos', 
                             'dow_sin', 'dow_cos',
                             'month_sin', 'month_cos']]
        
        # Delete intermediate dataframes
        del mains_df, app_df


        if debug:
            # plot the dtaset
            print("df_align with temporal features:")
            print(df_align.head())
            print(df_align.tail())
            print(f"\nTemporal feature ranges (sin/cos encoded):")
            print(f"  Minute sin: {df_align['minute_sin'].min():.3f} - {df_align['minute_sin'].max():.3f}")
            print(f"  Minute cos: {df_align['minute_cos'].min():.3f} - {df_align['minute_cos'].max():.3f}")
            print(f"  Hour sin: {df_align['hour_sin'].min():.3f} - {df_align['hour_sin'].max():.3f}")
            print(f"  Hour cos: {df_align['hour_cos'].min():.3f} - {df_align['hour_cos'].max():.3f}")
            print(f"  DOW sin: {df_align['dow_sin'].min():.3f} - {df_align['dow_sin'].max():.3f}")
            print(f"  DOW cos: {df_align['dow_cos'].min():.3f} - {df_align['dow_cos'].max():.3f}")
            print(f"  Month sin: {df_align['month_sin'].min():.3f} - {df_align['month_sin'].max():.3f}")
            print(f"  Month cos: {df_align['month_cos'].min():.3f} - {df_align['month_cos'].max():.3f}")
            # plt.plot(df_align['aggregate'].values)
            # plt.plot(df_align[appliance_name].values)
            # plt.savefig('{}.png'.format(appliance_name))
            # plt.show()
            test_len = int((len(df_align)/100)*testing_percent)

            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(111)

            # ax1.plot(df_align['aggregate'][-test_len:-1], color='#7f7f7f', linewidth=1.8)
            # ax1.plot(df_align[appliance_name][-test_len:-1], color='#d62728', linewidth=1.6)

            plt.subplot(211)
            plt.title(appliance_name)
            plt.plot(df_align['aggregate'][-test_len:])
            plt.yticks(np.linspace(0,5000,5,endpoint=True))

            plt.subplot(212)
            plt.plot(df_align[appliance_name][-test_len:])
            plt.yticks(np.linspace(0,5000,5,endpoint=True))
           
            
            # plt.subplots_adjust(bottom=0.2, right=0.7, top=0.9, hspace=0.3)
            plt.savefig('{}-_subplot.png'.format(args.appliance_name))
            # # ax1.plot(prediction,
            # #          color='#1f77b4',
            # #          #marker='o',
            # #          linewidth=1.5)
            # # plt.xticks([])
            # ax1.grid()
            # # ax1.set_title('Test results on {:}'.format(test_filename), fontsize=16, fontweight='bold', y=1.08)
            # ax1.set_ylabel(appliance_name)
            # ax1.legend(['aggregate', appliance_name],loc='upper left')

            # mng = plt.get_current_fig_manager()
            # #mng.resize(*mng.window.maxsize())
            # plt.savefig('{}.png'.format(args.appliance_name))

        # Normilization ----------------------------------------------------------------------------------------------
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']

        df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        df_align[appliance_name] = (df_align[appliance_name] - mean) / std

        # if h == params_appliance[appliance_name]['test_build']:
        #     # Test CSV
        #     df_align.to_csv(args.save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
        #     print("    Size of test set is {:.4f} M rows.".format(len(df_align) / 10 ** 6))
        #     continue

        train = pd.concat([train, df_align], ignore_index=True)
        del df_align

    # Crop dataset
    if training_building_percent is not 0:
        train.drop(train.index[-int((len(train)/100)*training_building_percent):], inplace=True)

    test_len = int((len(train)/100)*testing_percent)
    val_len = int((len(train)/100)*validation_percent)

    #Testing CSV
    test = train.tail(test_len)

    test.reset_index(drop=True, inplace=True)
    train.drop(train.index[-test_len:], inplace=True)
    test.to_csv(args.save_path + appliance_name + '_test_.csv', mode='w', index=False, header=True)


    # Validation CSV
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    # Validation CSV
    val.to_csv(args.save_path + appliance_name + '_validation_' + '.csv', mode='w', index=False, header=True)

    # Training CSV
    train.to_csv(args.save_path + appliance_name + '_training_.csv', mode='w', index=False, header=True)

    print("    Size of total training set is {:.4f} M rows.".format(len(train) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(len(val) / 10 ** 6))
    print("    Size of total testing set is {:.4f} M rows.".format(len(test) / 10 ** 6))
    del train, val, test


    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()