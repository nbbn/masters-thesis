# (c) Jakub Stepniak (https://github.com/nbbn/masters-thesis)

import pickle
import os
import pandas as pd
import datetime
import numpy as np
import random

import statsmodels.formula.api as sm_f
import statsmodels.stats.api as sms
import statsmodels.api as sm
import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta

import logging
from logging.config import dictConfig


class Magisterka:
    run_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

    def __init__(self):



        # self.logger init for logging to file/console
        self.logger = self.init_logger()
        self.logger.info('Magisterka init')

        self.only_gpw = False

        if self.only_gpw is True:
            self.list_of_determinants_and_vars = ['LEV_LTM',
                                                  'LEV_LTB',
                                                  'MB',
                                                  'Size',
                                                  'TANG',
                                                  'PROF',
                                                  'DivPayer',
                                                  'CFVol',
                                                  'LEV_IMed',
                                                  'gov_owned',
                                                  'growth_phase']
            self.list_of_float_determinants = ['LEV_LTM',
                                               'LEV_LTB',
                                               'MB',
                                               'Size',
                                               'TANG',
                                               'PROF',
                                               'CFVol',
                                               'LEV_IMed']
        else:
            self.list_of_determinants_and_vars = ['LEV_LTM',
                                                  'LEV_LTB',
                                                  'MB',
                                                  'Size',
                                                  'TANG',
                                                  'PROF',
                                                  'DivPayer',
                                                  'CFVol',
                                                  'LEV_IMed',
                                                  'growth_phase']
            self.list_of_float_determinants = ['LEV_LTM',
                                               'LEV_LTB',
                                               'MB',
                                               'Size',
                                               'TANG',
                                               'PROF',
                                               'CFVol',
                                               'LEV_IMed']




        self.run_id = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        self.raw_output_dir = 'output'
        self.model_output_dir = self.run_id + '_output'
        self.writer = pd.ExcelWriter(self.model_output_dir + '/data.xlsx')

        # set in list_of_files
        self.securities_filenames = None
        self.paths_to_securities_files = None

        # set in load_data
        self.data = None
        self.issues_data = {}
        self.ipo_dates = {}

        self._list_of_files()

        # set in load prices
        self.list_of_price_files = []
        self.paths_to_prices_files_with_path = []
        self.prices = {}

        # set in data_cleaning
        self.uber_data = pd.DataFrame()
        ###
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

        self.cycles = Magisterka.init_cycles()

    @staticmethod
    def init_logger():
        logging_config = dict(
            version=1,
            formatters={
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                }
            },
            handlers={
                'h': {'class': 'logging.StreamHandler',
                      'formatter': 'standard',
                      'level': logging.DEBUG},

                'file_handler': {
                    'level': logging.DEBUG,
                    'filename': 'log/log_' + Magisterka.run_id + '.log',
                    'class': 'logging.FileHandler',
                    'formatter': 'standard'
                }
            },
            loggers={
                '': {
                    'handlers': ['file_handler', 'h'],
                    'level': logging.INFO,
                    'propagate': True
                },
            }
        )
        dictConfig(logging_config)
        return logging.getLogger()

    @staticmethod
    def init_cycles():

        starts = ['1995-01', '1995-09', '1998-01', '1999-02', '2000-07', '2003-02', '2004-05', '2005-06', '2007-06',
                  '2008-11', '2010-09', '2013-02', '2014-02', '2015-03']

        ends = ['1995-08', '1997-12', '1999-01', '2000-06', '2003-01', '2004-04', '2005-05', '2007-05', '2008-10',
                '2010-08', '2013-01', '2014-01', '2015-02', '2017-02']

        for k, i in enumerate(starts):
            starts[k] = datetime.datetime.strptime(i, "%Y-%m").date()

        for k, i in enumerate(ends):
            date = datetime.datetime.strptime(i, "%Y-%m").date()
            date_max_days = (date + relativedelta(months=1, days=-1)).day
            ends[k] = datetime.datetime.strptime(i, "%Y-%m").date().replace(day=date_max_days)

        df = pd.DataFrame([starts, ends]).transpose()

        for i, x in df.iterrows():
            if i % 2 == 1:
                df.at[i, 'growth'] = 1
            else:
                df.at[i, 'growth'] = 0
        return df

    def _list_of_files(self):
        """Sets self.securities_filenames and self.paths_to_securities_files."""
        if self.only_gpw:
            path = 'NOTORIA_GPW_XLSX/'
            securities_list = os.listdir(path)
        else:
            path = 'NOTORIA_NC_XLSX/'
            securities_list = os.listdir(path)
        securities_list = [x for x in securities_list if not x.startswith('.')]
        securities_list.sort()
        self.securities_filenames = securities_list
        self.paths_to_securities_files = [path + x for x in securities_list]
        self.logger.debug('self.securities_filenames, n: {}, [0]: {}'.format(
            str(len(self.securities_filenames)),
            str(self.securities_filenames[0]))
        )
        self.logger.debug('self.paths_to_securities_files, n: {}, [0]: {}'.format(
            str(len(self.paths_to_securities_files)),
            str(self.paths_to_securities_files[0]))
        )

    def load_data(self, from_raw: bool = True, sample: float = 1, date: str = None) -> None:
        """It loads Notoria files or prepared pickle files and parse statements and IPO dates.
        At the end it runs load_prices.
        Input: Notoria files or pickles.
        Output:
        - self.data
          dict where keys are isins and values are DataFrames, that have as:
            index: Notoria codes and isin
            columns: ints 1-...
        - self.ipo_dates
          dict where keys are isins and values are datetime.date dates
        """
        if from_raw is False:
            self.data = self.read_from_pickle('{}/data_{}.pickle'.format(self.raw_output_dir, date))
            self.ipo_dates = self.read_from_pickle('{}/ipo_{}.pickle'.format(self.raw_output_dir, date))
            self.issues_data = self.read_from_pickle('{}/issues_{}.pickle'.format(self.raw_output_dir, date))
        else:
            sample_size = round(len(self.paths_to_securities_files) * sample)
            self.logger.debug('data_loader… number of secs in sample: {}'.format(sample_size))
            self.data = {}
            self.paths_to_securities_files = random.sample(self.paths_to_securities_files, sample_size)
            for i_file, sec_file_path in enumerate(self.paths_to_securities_files):
                self.logger.info('start: {}'.format(sec_file_path))
                # read info sheet from excel file
                h = pd.read_excel('file:' + sec_file_path, sheetname='Info', header=None)

                isin_cell = h.iloc[17, 4]
                industry_cell = h.iloc[20, 4]
                ipo_cell = h.iloc[1, 1]
                if pd.isnull(isin_cell) or pd.isnull(ipo_cell) or pd.isnull(industry_cell):
                    self.logger.info('without isin/sector/ipo: {}'.format(sec_file_path))
                    continue
                isin = isin_cell.lower()
                try:
                    self.ipo_dates[isin] = datetime.datetime.strptime(ipo_cell, "%Y-%m-%d").date()
                except ValueError:
                    self.logger.info('invalid ipo date: {}'.format(sec_file_path))
                    continue

                # shareholders (look for skarb panstwa)
                shareholders = pd.read_excel('file:' + sec_file_path, sheetname='Shareholders', header=None)
                is_owned_by_gov = bool(shareholders.iloc[:, 0].isin(['Skarb Państwa']).any())

                # ISSUES
                issues = pd.read_excel('file:' + sec_file_path, sheetname='Issues', header=None)
                # select data about issues
                issues_to_save = issues.iloc[1:, [5, 1, 8, 9, 7]]
                issues_to_save.columns = ['number_of_shares', 'date_wza', 'date_reg', 'date_gielda', 'date_knf']
                issues_to_save.loc[:, 'number_of_shares'] = pd.to_numeric(issues_to_save.loc[:, 'number_of_shares'])
                for i in ['date_wza', 'date_reg', 'date_gielda', 'date_knf']:
                    try:
                        issues_to_save.loc[issues_to_save.loc[:, i].notnull(), i] = \
                            issues_to_save.loc[issues_to_save.loc[:, i].notnull(), i].map(
                                lambda m: datetime.datetime.strptime(m, "%Y-%m-%d").date())
                    except (TypeError, ValueError):
                        for iii, x in issues_to_save.loc[:, i].iteritems():
                            try:
                                issues_to_save.loc[iii, i] = datetime.datetime.strptime(x, "%Y-%m-%d").date()
                            except (TypeError, ValueError):
                                issues_to_save.loc[iii, i] = np.nan
                self.issues_data[isin] = issues_to_save

                self.logger.debug(f'adding to ipo_dates: {isin} {self.ipo_dates[isin]}')

                # read YC sheet from excel file
                h = pd.read_excel('file:' + sec_file_path, sheetname='YC', header=None, index_col=0)
                if len(h.columns) <= 3:
                    # in tab YC there is no columns, so I assume that company has no consolidated
                    #  financial statements and I skip to the next company
                    self.logger.warning('file without valid dates in statements: {} {}'.format(sec_file_path, isin))
                    del self.ipo_dates[isin]
                    continue
                h = h.loc[~h.index.duplicated(keep='first'), :]
                h.loc['industry', :] = industry_cell
                h.loc['isin', :] = isin
                if is_owned_by_gov is True:
                    h.loc['gov_owned', :] = 1
                else:
                    h.loc['gov_owned', :] = 0

                self.data[isin] = h
            self.logger.info(f'secs with valid  ipo/isin/industry: {len(self.data)}')
            self.save_to_pickle(self.data, 'data')
            self.save_to_pickle(self.ipo_dates, 'ipo')
            self.save_to_pickle(self.issues_data, 'issues')

    def load_prices(self, from_raw: bool = True, date: str = None) -> None:
        """It loads data from GPWInfostrefa files and save it to self.prices.
        Input: GPWInfostrefa files or pickles
        Output:
        self.prices
          dict where keys are isins and values are DataFrames with columns: "eod_price"
          indicies: 'date' str with data in iso format (datetime64)
        """
        if from_raw is False:
            self.prices = self.read_from_pickle(f'{self.raw_output_dir}/prices_{date}.pickle')
        else:
            self.logger.debug('load_prices…')
            path = 'gpwinfostrefa/'
            prices = os.listdir(path)
            prices = [x for x in prices if not x.startswith('.')]
            prices.sort()
            self.list_of_price_files = prices
            self.paths_to_prices_files_with_path = [path + x for x in prices]
            self.logger.debug('self.prices, n: {}, [0]: {}'.format(
                str(len(self.list_of_price_files)),
                str(self.list_of_price_files[0]))
            )
            self.logger.debug('self.paths_to_prices_files, n: {}, [0]: {}'.format(
                str(len(self.paths_to_prices_files_with_path)),
                str(self.paths_to_prices_files_with_path[0]))
            )
            self.prices = {}
            for i_price_file, price_file in enumerate(self.paths_to_prices_files_with_path):
                isin = price_file[price_file.index('/') + 1:price_file.index('_')].lower()
                try:
                    single_sec = pd.read_excel(price_file, 'Sheet0', header=0, index_col=0, parse_cols=[1, 8])
                    single_sec.set_index(pd.DatetimeIndex(single_sec.index), inplace=True)
                    single_sec.columns = ['eod_price']
                    single_sec.index.rename('date', inplace=True)
                    self.prices[isin] = single_sec
                except:
                    self.logger.warning(f'sec without prices: {isin}')
            self.save_to_pickle(self.prices, name='prices')

    def data_preparation(self) -> None:
        """It reads pickled dataframes, then it excludes banking and insurances.
        After that it deletes empty rows/cols in statements."""
        self.logger.info('data cleaning')
        self.logger.info('num of secs: {}, num of ipo_dates: {}, num of secs with prices: {}'.format(
            len(self.data),
            len(self.ipo_dates),
            len(self.prices)
        ))
        excluded = []
        excluded = [i.lower() for i in excluded]
        self.logger.info(f'number of excluded: {len(excluded)}')
        for i in excluded:
            self.data.pop(i)
        for s in self.data:
            # columns with empty assets sum (empty columns and other situations)
            self.data[s].dropna(axis='columns', how='any', subset=['A_0'], inplace=True)
            # columns with descriptions (polish and english names of values)
            self.data[s].drop(self.data[s].columns[[0, 1]], inplace=True, axis=1)

        self.logger.info(f'number of secs after cleaning: {len(self.data)}')
        data_list = [k for k in self.data.values()]
        self.uber_data = pd.concat(data_list, ignore_index=True, axis=1)
        self.uber_data = self.uber_data.transpose()
        self.uber_data = self.uber_data.loc[:, pd.notnull(self.uber_data.columns)]

    def add_vars(self, dump=False) -> None:
        """It makes place for derived values and try to calculate some of them."""
        self.logger.info('add_vars')

        # delete finance and insurance companies
        self.uber_data = self.uber_data.loc[
                         (self.uber_data.loc[:, 'industry'] != 'banki')
                         & (self.uber_data.loc[:, 'industry'] != 'ubezpieczenia'), :]

        self.uber_data.loc[:, 'D_BZ_date'] = self.uber_data.loc[:, 'D_BZ'].map(
            lambda m: datetime.datetime.strptime(m, "%Y-%m-%d").date())
        self.uber_data.loc[:, 'year'] = self.uber_data.loc[:, 'D_BZ_date'].map(lambda n: n.year)

        for i, x in self.uber_data.iterrows():
            statement_date = x.at['D_BZ_date']
            reference_date = statement_date
            if x.at['isin'] not in self.prices.keys():
                self.logger.warning('isin in data not in prices: {}'.format(x.at['isin']))
                continue
            # I don't trust Notoria, so I will check on my own in GPWinfostrefa for prices for debug
            before_ipo_date_check = False
            if reference_date <= self.ipo_dates[x.at['isin']]:
                before_ipo_date_check = True

            # add market price for balance sheet date
            while True:
                try:
                    self.uber_data.loc[i, 'market_price'] = self.prices[x.at['isin']].loc[reference_date, 'eod_price']
                    self.uber_data.loc[i, 'market_price_date'] = reference_date
                    break
                except KeyError:
                    reference_date = reference_date - datetime.timedelta(days=1)
                    distance = statement_date - reference_date
                    if distance >= datetime.timedelta(days=10):
                        if before_ipo_date_check is False:
                            self.logger.warning('achtung! no price in last {}, isin: {}, {}.'.format(
                                distance, x.at['isin'], statement_date
                            ))
                        break

            # find price at the end of year for market cap calc
            # d = 31
            # while d > 25:
            #     date = datetime.date(year=x.at['year'], month=12, day=d)
            #     try:
            #         self.uber_data.loc[i, 'price_end_of_year'] = self.prices[x.at['isin']].loc[date, 'eod_price']
            #     except KeyError:
            #         d = d - 1

        for i, x in self.uber_data.iterrows():
            self.uber_data.at[i, 'better_num_of_shares'] = self.get_number_of_shares(
                isin=x['isin'],
                date=x['D_BZ_date']
            )

        self.uber_data.loc[:, 'better_num_of_shares'] = pd.to_numeric(self.uber_data.loc[:, 'better_num_of_shares'])
        self.uber_data.loc[:, 'combined_num_of_shares'] = self.uber_data.loc[:, 'better_num_of_shares'] / 1000

        self.uber_data.loc[:, 'A_0'] = pd.to_numeric(self.uber_data.loc[:, 'A_0'])  # many calcs
        self.uber_data.loc[:, 'PP_1'] = pd.to_numeric(self.uber_data.loc[:, 'PP_1'])  # equity calc
        self.uber_data.loc[:, 'PP_2'] = pd.to_numeric(self.uber_data.loc[:, 'PP_2'])  # LEV_LTM calc
        self.uber_data.loc[:, 'P_8'] = pd.to_numeric(self.uber_data.loc[:, 'P_8'])  # equity calc
        self.uber_data.loc[:, 'WO_0'] = pd.to_numeric(self.uber_data.loc[:, 'WO_0'])  # PROF calc
        self.uber_data.loc[:, 'R[AM_0]'] = pd.to_numeric(self.uber_data.loc[:, 'R[AM_0]'])  # PROF calc
        self.uber_data.loc[:, 'AA_1'] = pd.to_numeric(self.uber_data.loc[:, 'AA_1'])  # TANG calc
        self.uber_data.loc[:, 'market_price'] = pd.to_numeric(self.uber_data.loc[:, 'market_price'])

        # market value = market_price * combined_num_of_shares (ilość akcji w tysiącach) (uzyskana wartość w tysiącach)
        self.uber_data.loc[:, 'market_value'] = self.uber_data.loc[:, 'market_price'] * \
            self.uber_data.loc[:, 'combined_num_of_shares']
        self.uber_data.loc[:, 'market_value'] = pd.to_numeric(self.uber_data.loc[:, 'market_value'])

        # PP_2 / A_0 (zobowiązania długoterminowe / suma aktywów)
        self.uber_data.loc[:, 'LEV_LTB'] = self.uber_data.loc[:, 'PP_2'] / self.uber_data.loc[:, 'A_0']
        self.uber_data.loc[:, 'LEV_LTB'] = pd.to_numeric(self.uber_data.loc[:, 'LEV_LTB'])

        # PP_2 /(PP_2+market_value) zobowiązania_długoterminowe / (zobowiązania_długoterminowe+wartość rynkowa)
        self.uber_data.loc[:, 'LEV_LTM'] = (self.uber_data.loc[:, 'PP_2']) / \
                                           (self.uber_data.loc[:, 'PP_2'] + self.uber_data.loc[:, 'market_value'])
        self.uber_data.loc[:, 'LEV_LTM'] = pd.to_numeric(self.uber_data.loc[:, 'LEV_LTM'])

        # ln (A_0) ln(suma aktywów)
        self.uber_data.loc[:, 'Size'] = np.log(self.uber_data.loc[:, 'A_0'].tolist())
        self.uber_data.loc[:, 'Size'] = pd.to_numeric(self.uber_data.loc[:, 'Size'])

        self.uber_data.loc[:, 'equity'] = (self.uber_data.loc[:, 'PP_1'] + self.uber_data.loc[:, 'P_8'])
        self.uber_data.loc[:, 'equity'] = pd.to_numeric(self.uber_data.loc[:, 'equity'])

        # market_value / (PP_1 + P_8)
        # (wartość rynkowa w tys / (Kapitał własny udziałowców podmiotu dominującego + Udziały niekontrolujące)
        # (different procedures)
        self.uber_data.loc[:, 'MB'] = self.uber_data.loc[:, 'market_value'] / self.uber_data.loc[:, 'equity']

        self.uber_data.loc[:, 'MB'] = pd.to_numeric(self.uber_data.loc[:, 'MB'])

        # (WO_0+R[AM_0]) / A_0 ((Zysk/strata z działalności operacyjnej+amortyzacja) / suma aktywów)
        self.uber_data.loc[:, 'PROF'] = (self.uber_data.loc[:, 'WO_0'] + self.uber_data.loc[:, 'R[AM_0]'])\
            / self.uber_data.loc[:, 'A_0']
        self.uber_data.loc[:, 'PROF'] = pd.to_numeric(self.uber_data.loc[:, 'PROF'])

        # AA_1 / A_0 (Aktywa trwałe / suma aktywów)
        self.uber_data.loc[:, 'TANG'] = self.uber_data.loc[:, 'AA_1'] / self.uber_data.loc[:, 'A_0']
        self.uber_data.loc[:, 'TANG'] = pd.to_numeric(self.uber_data.loc[:, 'TANG'])

        self.uber_data.loc[self.uber_data.loc[:, 'CFC_7'] < 0, 'DivPayer'] = 1
        self.uber_data.loc[self.uber_data.loc[:, 'DivPayer'] != 1, 'DivPayer'] = 0
        self.uber_data.loc[:, 'DivPayer'] = pd.to_numeric(self.uber_data.loc[:, 'DivPayer'])

        self.uber_data.loc[self.uber_data.loc[:, 'gov_owned'] != 1, 'gov_owned'] = 0
        self.uber_data.loc[:, 'gov_owned'] = pd.to_numeric(self.uber_data.loc[:, 'gov_owned'])

        self.uber_data.loc[:, 'growth_phase'] = self.uber_data.loc[:, 'D_BZ_date'].map(lambda n: self.get_phase(n))
        self.uber_data.loc[:, 'growth_phase'] = pd.to_numeric(self.uber_data.loc[:, 'growth_phase'])

        self.uber_data.loc[:, 'LEV_IMed'] = np.nan
        group_criterion = 'industry'
        for i, x in self.uber_data.iterrows():
            ss = x.at[group_criterion]
            for ii in self.uber_data.loc[:, group_criterion].unique():
                if ss == ii:
                    self.uber_data.loc[i, 'LEV_IMed'] = self.uber_data.loc[
                        self.uber_data.loc[:, group_criterion] == ii, 'LEV_LTB'].median()
                    continue
        self.uber_data.loc[:, 'LEV_IMed'] = pd.to_numeric(self.uber_data.loc[:, 'LEV_IMed'])

        # CFVol
        self.uber_data.loc[:, 'CFVol'] = np.nan
        for i, x in self.uber_data.iterrows():
            older = self.uber_data.loc[(self.uber_data.loc[:, 'isin'] == x.at['isin']) &
                                       (self.uber_data.loc[:, 'D_BZ_date'] < x.at['D_BZ_date']) &
                                       (self.uber_data.loc[:, 'WO_0'].notnull()), :]
            if len(older) >= 3:
                self.uber_data.at[i, 'CFVol'] = older.loc[:, 'WO_0'].std()
        self.uber_data.loc[:, 'CFVol'] = pd.to_numeric(self.uber_data.loc[:, 'CFVol'])

        self.uber_data.set_index(['isin', 'D_BZ'], inplace=True, drop=False)

        if dump is True:
            self.save_to_pickle(self.uber_data, 'uber_data')

    def save_to_pickle(self, variable, name: str) -> None:
        try:
            pickle.dump(variable,
                        open('{}/{}_{}.pickle'.format(self.raw_output_dir, name, self.run_id),
                             'wb'))
        except pickle.PickleError:
            raise Exception('picking error')

    def get_number_of_shares(self, isin, date):
        w = np.nan
        # self.logger.info('get_number_of_shares {} {}'.format(isin, date))

        for i, x in self.issues_data[isin].iterrows():
            dates = x[['date_wza', 'date_reg', 'date_knf', 'date_gielda']]
            # skip header or rows without dates
            if dates.isnull().values.all():
                self.logger.debug('dates are null')
                continue
            if x.at['number_of_shares'] == np.nan:
                self.logger.warning('number_of_shares is null')
                continue

            ref_date = dates[dates.notnull()].min()
            if ref_date <= date:
                w = x.at['number_of_shares']
            # it will continue for loop until next row with higher date,
            # if here is no row like that, it will try to return w var.
            else:
                if w == np.nan:
                    self.logger.warning('all dates after: not able to get num of shares for {} {}'.format(isin, date))
                return w
        # for ended without return (no line after date)
        if w == np.nan:
            self.logger.warning('no dates at all: not able to get num of shares for {} {}'.format(isin, date))
        return w

    @staticmethod
    def read_from_pickle(fn: str):
        try:
            to_return = pickle.load(open(fn, 'rb'))
        except pickle.PickleError:
            raise Exception('picking error')
        else:
            return to_return

    @classmethod
    def simple_stat(cls, array: iter, desc='stats') -> None:
        try:
            print(
                '{7:17}:  <{0:12.2f}..{2:12.2f}..{3:12.2f}..{4:12.2f}..{1:12.2f}>'
                ' ({5:12.2f}) std:{6:10.2f}, uniq:{9:8} count:{8:8}'.format(
                    array.min(),
                    array.max(),
                    array.quantile(0.25),
                    array.quantile(0.5),
                    array.quantile(0.75),
                    array.mean(),
                    array.std(),
                    desc,
                    array.count(),
                    len(array.unique()),
                ))
        except ValueError:
            print('{:15}: invalid data - value error'.format(desc), '')
        except TypeError:
            print('{:15}: invalid data - type error'.format(desc), '')

    def show_stats_of_vars(self) -> None:
        print('D_BZ: {} - {}'.format(self.uber_data.loc[:, 'D_BZ'].min(), self.uber_data.loc[:, 'D_BZ'].max()))
        print('D_BZ_date: {} - {}'.format(self.uber_data.loc[:, 'D_BZ_date'].min(),
                                          self.uber_data.loc[:, 'D_BZ_date'].max()))
        print('year: {} - {}'.format(self.uber_data.loc[:, 'year'].min(), self.uber_data.loc[:, 'year'].max()))
        print('number of isins: {}'.format(len(self.uber_data.loc[:, 'isin'].unique())))
        print(
            '{7:17}:  <{0:12}..{2:12}..{3:12}..{4:12}..{1:12}>'
            ' ({5:12}) std:{6:10}, uniq:{9:8} count:{8:8}'.format(
                'min', 'max', 'q0.25', 'q0.5', 'q0.75', 'mean', 'std', 'description', 'count', 'unique'
            ))
        self.simple_stat(self.uber_data.loc[:, 'market_price'], 'market_price')
        self.simple_stat(self.uber_data.loc[:, 'market_value'], 'market_value')
        self.simple_stat(self.uber_data.loc[:, 'PP_2'], 'PP_2')
        self.simple_stat(self.uber_data.loc[:, 'A_0'], 'A_0')
        self.simple_stat(self.uber_data.loc[:, 'PP_1'], 'PP_1')
        self.simple_stat(self.uber_data.loc[:, 'P_8'], 'P_8')
        self.simple_stat(self.uber_data.loc[:, 'WO_0'], 'WO_0')
        self.simple_stat(self.uber_data.loc[:, 'R[AM_0]'], 'R[AM_0]')
        self.simple_stat(self.uber_data.loc[:, 'AA_1'], 'AA_1')
        self.simple_stat(self.uber_data.loc[:, 'CFC_7'], 'CFC_7')
        print(' ')
        self.simple_stat(self.uber_data.loc[:, 'LEV_LTB'], 'LEV_LTB')
        self.simple_stat(self.uber_data.loc[:, 'LEV_LTM'], 'LEV_LTM')
        self.simple_stat(self.uber_data.loc[:, 'Size'], 'Size')
        self.simple_stat(self.uber_data.loc[:, 'MB'], 'MB')
        self.simple_stat(self.uber_data.loc[:, 'PROF'], 'PROF')
        self.simple_stat(self.uber_data.loc[:, 'TANG'], 'TANG')
        self.simple_stat(self.uber_data.loc[:, 'LEV_IMed'], 'LEV_IMed')
        self.simple_stat(self.uber_data.loc[:, 'DivPayer'], 'DivPayer')
        self.simple_stat(self.uber_data.loc[:, 'CFVol'], 'CFVol')

    def make_model_ols(self, model_id: str) -> None:

        results = sm_f.ols(formula="LEV_LTM ~ MB + Size + TANG + PROF + DivPayer + CFVol + LEV_IMed + growth_phase",
                           data=self.uber_data).fit()
        print(results)
        print(results.pvalues)
        with open('{}/model.txt'.format(self.model_output_dir), "w") as text_file:
            text_file.write(str(results.summary()))
        self.logger.info(results.summary())

        fig = plt.figure(figsize=(12, 8))
        sm.graphics.plot_partregress_grid(results, fig=fig)
        fig.savefig('{}/{}_{}_partregress.png'.format(self.model_output_dir, self.run_id, model_id))

        fig = plt.figure(figsize=(12, 8))
        sm.graphics.plot_ccpr_grid(results, fig=fig)
        fig.savefig('{}/{}_{}_ccpr.png'.format(self.model_output_dir, self.run_id, model_id))

        fig, ax = plt.subplots(figsize=(12, 8))
        sm.graphics.influence_plot(results, ax=ax, criterion="cooks")
        fig.savefig('{}/{}_{}_influence.png'.format(self.model_output_dir, self.run_id, model_id))

        plt.figure(figsize=(12, 8))
        res = results.resid
        sm.qqplot(res)
        plt.savefig('{}/{}_{}_qqplot.png'.format(self.model_output_dir, self.run_id, model_id))

        # plt.pcolor(self.for_correlation)
        # plt.yticks(np.arange(0.5, len(self.for_correlation.index), 1), self.for_correlation.index)
        # plt.xticks(np.arange(0.5, len(self.for_correlation.columns), 1), self.for_correlation.columns)
        # plt.savefig('{}/{}_{}_heatmap_of_covar.png'.format(self.model_output_dir, self.run_id, model_id))

    def dump_standard_stats(self, sheet_name='s'):
        index = ['D_BZ',
                 'D_BZ_date',
                 'year',
                 'LEV_LTB',
                 'LEV_LTM',
                 'Size',
                 'MB',
                 'PROF',
                 'TANG',
                 'LEV_IMed',
                 'DivPayer',
                 'CFVol',
                 'growth_phase',
                 'better_num_of_shares',
                 'combined_num_of_shares',
                 'isin',
                 'market_price',
                 'market_value',
                 'equity',
                 'PP_2',
                 'A_0',
                 'PP_1',
                 'P_8',
                 'WO_0',
                 'R[AM_0]',
                 'AA_1',
                 'O_1',  # number_of_shares
                 'CFC_7',
                 ]
        columns = ['min', 'q0.25', 'q0.5', 'q0.75', 'max', 'mean', 'std', 'unique', 'count', ]
        stats = pd.DataFrame(columns=columns, index=index)
        for i, x in stats.iterrows():
            for ii in stats.columns:
                try:
                    if ii == 'min':
                        stats.at[i, ii] = self.uber_data.loc[:, i].min()
                    elif ii == 'q0.25':
                        stats.at[i, ii] = self.uber_data.loc[:, i].quantile(0.25)
                    elif ii == 'q0.5':
                        stats.at[i, ii] = self.uber_data.loc[:, i].median()
                    elif ii == 'q0.75':
                        stats.at[i, ii] = self.uber_data.loc[:, i].quantile(0.75)
                    elif ii == 'max':
                        stats.at[i, ii] = self.uber_data.loc[:, i].max()
                    elif ii == 'mean':
                        stats.at[i, ii] = self.uber_data.loc[:, i].mean()
                    elif ii == 'std':
                        stats.at[i, ii] = self.uber_data.loc[:, i].std()
                    elif ii == 'unique':
                        stats.at[i, ii] = len(self.uber_data.loc[:, i].unique())
                    elif ii == 'count':
                        stats.at[i, ii] = self.uber_data.loc[:, i].count()
                except TypeError:
                    stats.at[i, ii] = np.nan

        stats.to_excel(self.writer, sheet_name='stats_{}'.format(sheet_name))
        self.uber_data.to_excel(self.writer, sheet_name='data_{}'.format(sheet_name))

    def delete_quantile(self, l, limit=0.005):
        for i in l:
            self.logger.info('delete_quantile {}, {}'.format(i, self.uber_data.loc[:, i].quantile(limit)))
            self.uber_data.loc[
                (self.uber_data.loc[:, i] < self.uber_data.loc[:, i].quantile(limit))
                | (self.uber_data.loc[:, i] > self.uber_data.loc[:, i].quantile(1-limit)),
                i] = np.nan
        self.uber_data.dropna(axis='rows', how='any',
                              subset=l,
                              inplace=True)

    def model_lemmon_graph(self, l_types=False):   # ['LEV_LTB', 'LEV_LTM']
        """Prepare and add to output excel file sheets with data needed to
        plot (Fig.1) that was in Lemmon et al. (2008)."""

        if type(l_types) == str:
            l_types = [l_types]

        for l in l_types:
            # portfolios' names
            index = ['vhigh', 'high', 'med', 'low']
            # numbers 0-x where x is number of years in research-1 (-6 because of lack of obs)
            columns = list(range(8))
            print(columns)
            # create output DF
            lemmon_graph = pd.DataFrame(index=index, columns=columns)
            lemmon_graph_limit_up = pd.DataFrame(index=index, columns=columns)
            lemmon_graph_limit_down = pd.DataFrame(index=index, columns=columns)
            lemmon_graph_n = pd.DataFrame(index=index, columns=columns)

            # list of calendar years in research period, sorted
            years_range = sorted(list(range(
                self.uber_data.loc[:, 'D_BZ_date'].max().year, self.uber_data.loc[:, 'D_BZ_date'].min().year - 1, -1)
            ))
            for calendar_year in years_range:
                for relative_year in lemmon_graph.columns:
                    for portfolio in ['vhigh', 'high', 'med', 'low']:
                        self.logger.debug('lemon prep for {} {} {}'.format(calendar_year, relative_year, portfolio))
                        if portfolio == 'vhigh':
                            threshold_up = 2
                            threshold_dn = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.75)
                        elif portfolio == 'high':
                            threshold_up = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.75)
                            threshold_dn = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.5)
                        elif portfolio == 'med':
                            threshold_up = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.5)
                            threshold_dn = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.25)
                        elif portfolio == 'low':
                            threshold_up = self.uber_data.loc[
                                self.uber_data.loc[:, 'year'] == calendar_year, l].quantile(0.25)
                            threshold_dn = -1
                        else:
                            raise Exception
                        if relative_year == 0:
                            self.uber_data.loc[
                                (self.uber_data.loc[:, l] < threshold_up) &
                                (self.uber_data.loc[:, l] > threshold_dn) &
                                (self.uber_data.loc[:, 'year'] == calendar_year),
                                f'{calendar_year}_{relative_year}'] = portfolio
                        else:
                            self.uber_data.loc[:, f'{calendar_year}_{relative_year}'] = np.nan
                            for i, x in self.uber_data.loc[
                                        self.uber_data.loc[:, f'{calendar_year}_{relative_year-1}'].notnull(), :
                                        ].iterrows():
                                self.uber_data.loc[
                                    (self.uber_data.loc[:, 'isin'] == x['isin']) &
                                    (self.uber_data.loc[:, 'year'] == x['year'] + 1),
                                    f'{calendar_year}_{relative_year}'] = \
                                    x[f'{calendar_year}_{relative_year-1}']

            for relative in lemmon_graph.columns:
                for portfolio, r in lemmon_graph.iterrows():
                    for_mean = []
                    for i in years_range:
                        try:
                            for_mean += self.uber_data.loc[
                                self.uber_data.loc[:, f'{i}_{relative}'] == portfolio, l].tolist()
                        except TypeError:
                            self.logger.info(f'type error in lemmon: i: {i} p: {portfolio} r: {relative}.')
                            pass
                    pd_for_mean = pd.Series(for_mean)
                    lemmon_graph.at[portfolio, relative] = pd_for_mean.mean()
                    lemmon_graph_n.at[portfolio, relative] = pd_for_mean.count()
                    lemmon_graph_limit_up.at[portfolio, relative] = sms.DescrStatsW(pd_for_mean).tconfint_mean()[1]
                    lemmon_graph_limit_down.at[portfolio, relative] = sms.DescrStatsW(pd_for_mean).tconfint_mean()[0]

            results = {'main_data': lemmon_graph,
                    'amount': lemmon_graph_n,
                    'limit_up': lemmon_graph_limit_up,
                    'limit_down': lemmon_graph_limit_down
                    }
            return results

    def describe_data(self, name=''):
        # change of something in time
        a = self.uber_data.groupby(by='year')['LEV_LTB'].mean()
        # self.uber_data.plot(kind='scatter', subplots=True)
        # plt.savefig('{}/{}_{}.png'.format(self.model_output_dir, self.run_id, f'change_lev_ltb_{name}'))

        # histogram of variables in time
        # self.uber_data.hist()

        # plt.savefig('{}/{}_{}.png'.format(self.model_output_dir, self.run_id, f'hist_{name}'))

    def get_phase(self, date):
        a_before_six = date + relativedelta(months=-6)
        inter = self.cycles.loc[(self.cycles.loc[:, 0] < a_before_six) & (self.cycles.loc[:, 1] >= a_before_six), :]
        growth_phase = inter.iloc[0, 2]
        return growth_phase


def main(mode=None):
    # init
    b = Magisterka()
    saved_date = '20170621_2343'
    # OLD (BEFORE UPDATE): official data for NC - 20170527_1255, wse - 20170527_1209
    # NEW (BEFORE UPDATE): official data for NC - 20170622_0145, wse - 20170621_2343

    # load data and prices from raw files or cached pickled version
    b.load_data(from_raw=False, sample=1, date=saved_date)
    b.load_prices(from_raw=False, date=saved_date)

    stage_2_from_cache = True
    if stage_2_from_cache is False:
        b.logger.info('stage 2 execution')
        # data preparation merge single files and delete unneeded elements
        b.data_preparation()

        # calculate variables and save them for future use
        dump_uber_data = True
        b.add_vars(dump=dump_uber_data)
    else:
        b.logger.info('stage 2 cache loading')
        b.uber_data = b.read_from_pickle(f'output/uber_data_{saved_date}.pickle')

    stage_3_from_cache = True
    if stage_3_from_cache is False:
        b.logger.info('stage 3 execution')
        # dump stats with uncleaned variables to spreadsheet file
        b.dump_standard_stats(sheet_name='b_clean')
        b.describe_data(name='b_clean')

        # delete statements without source data (sum of assets) and all independent variables
        b.uber_data.dropna(axis='rows', how='any',
                           subset=b.list_of_determinants_and_vars,
                           inplace=True)

        # dump stats with with simple cleaned variables to spreadsheet file
        b.dump_standard_stats(sheet_name='clean')

        # select data for correlation matrix and dump it to spreadsheet file (after simple clean)
        b.for_correlation = b.uber_data.loc[:, b.list_of_determinants_and_vars]
        b.for_correlation.corr().to_excel(b.writer, sheet_name='covar_matrix')

        b.dump_standard_stats(sheet_name='t-0')
        # assumptions about variables (research logic and sophisticated/statistical cleaning
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'LEV_LTB'] <= 1, :]  # Lemmon et al. (2008) //dziala
        b.dump_standard_stats(sheet_name='t-1')
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'LEV_LTB'] >= 0, :]  # Lemmon et al. (2008)
        b.dump_standard_stats(sheet_name='t-2')
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'LEV_LTM'] <= 1, :]  # Lemmon et al. (2008)
        b.dump_standard_stats(sheet_name='t-3')
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'LEV_LTM'] >= 0, :]  # Lemmon et al. (2008)
        b.dump_standard_stats(sheet_name='t-4')
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'O_7'] == 'PLN', :]  # for correct calc of ratios //dziala
        b.dump_standard_stats(sheet_name='t-5')
        b.uber_data = b.uber_data.loc[b.uber_data.loc[:, 'O_2'] == 12, :]  # research logic
        b.dump_standard_stats(sheet_name='t-6')
        # Lemmon et al. (2008) - delete top and bottom 0.5%
        b.delete_quantile(b.list_of_float_determinants,
                          limit=0.02)

        b.logger.info('stage 3 end')
        save_stage_3_cache = True
        if save_stage_3_cache:
            b.save_to_pickle(b.uber_data, 'stage_3')
    else:
        b.logger.info('stage 3 cache loading')
        b.uber_data = b.read_from_pickle(f'output/stage_3_{saved_date}.pickle')
    b.describe_data(name='after_clean_assumptions')

    # exit()

    if mode == 'console':
        return b

    # replicate Lemmon et al. (2008) Fig.1
    r = b.model_lemmon_graph(l_types='LEV_LTB')
    r['main_data'].to_excel(b.writer, sheet_name='main_data')
    r['amount'].to_excel(b.writer, sheet_name='amount')
    r['limit_up'].to_excel(b.writer, sheet_name='limit_up')
    r['limit_down'].to_excel(b.writer, sheet_name='limit_down')

    # dump stats with statistical cleaned variables to spreadsheet file
    b.dump_standard_stats(sheet_name='clean_assumptions')

    # select data for correlation matrix and dump it to spreadsheet file (after statistical clean)
    b.for_correlation = b.uber_data.loc[:, b.list_of_determinants_and_vars]
    b.for_correlation.corr().to_excel(b.writer, sheet_name='covar_matrix_assumptions')

    b.make_model_ols('1_ols')

    # write spreadsheet/csv files
    b.writer.save()
    b.writer.close()
    b.uber_data.to_csv(b.model_output_dir + '/uber_data.csv')

if __name__ == '__main__':
    main()
