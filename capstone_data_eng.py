# -*- coding: utf-8 -*-
"""

data engineering functions

"""

import sqlite3
import sys

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pylab as plt
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

from auxiliary import *

def list_channels_to_df(portfolio, channel_types=['web', 'email', 'mobile', 'social'], debug=False):
    """
    extracts information from the dictionary located in the channels column of the portfolio database to separate columns

    :param portfolio:
    :param channel_types:
    :param debug:
    :return:
    """
    #

    # initialize
    for channel in channel_types:
        portfolio[channel] = False

    # assign values
    for ind in portfolio.index:
        for channel in channel_types:
            if debug:
                print(portfolio.loc[ind, 'channels'])
            if channel in portfolio.loc[ind, 'channels']:
                portfolio.loc[ind, channel] = True
                if debug:
                    print(channel, portfolio[channel])
    return (portfolio)


def load_portfolio(debug=False, location='colab'):
    """
    loads and processes the portfolio database

    :param debug: print debug information
    :param location: portfolio.json location (Google Colab or local)
    :return:
    """

    if location == 'colab':
        portfolio = pd.read_json(r'/content/gdrive/My Drive/UD/Sbux/data/portfolio.json', orient='records', lines=True)
    else:
        portfolio = pd.read_json(r'portfolio.json', orient='records', lines=True)

    # extracts channel (web, email, ...) information
    portfolio = list_channels_to_df(portfolio);
    portfolio = portfolio.drop(columns=['channels'])
    if debug:
        print('portfolio.shape:', portfolio.shape)

    assert portfolio.shape[0] == 10, 'wrong portfolio size'
    n_offers = np.unique(portfolio['id']).shape[0]
    assert n_offers == 10, 'wrong portfolio size'
    portfolio.rename(columns={'id': 'offer', 'reward': 'offer_reward'}, inplace=True)
    portfolio['duration_hours'] = 24. * portfolio['duration']

    return (portfolio)


def load_transcript(debug=False, location='colab'):
    """
    loads and processes the transcript database

    :param debug: print debug information
    :param location: portfolio.json location (Google Colab or local)
    :return:
    """

    if location == 'colab':
        transcript = pd.read_json(r'/content/gdrive/My Drive/UD/Sbux/data/transcript.json', orient='records',
                                  lines=True)
    else:
        transcript = pd.read_json(r'transcript.json', orient='records', lines=True)

    # extract offers + transactions
    if debug:
        print('transcript.shape:', transcript.shape)

    # extract if/when the offer was viewed etc
    values = (pd.DataFrame(transcript['value'].values.tolist()))  # .stack()
    values.columns = ['transaction', 'offer', 'offer viewed', 'reward']

    transcript = pd.concat([transcript, values], axis=1)
    transcript['offer'] = transcript['offer'].fillna(transcript['offer viewed'])
    transcript = transcript.drop(columns=['offer viewed'])

    if debug:
        print('transcript.shape:', transcript.shape)

    n_persons_transcript = np.unique(transcript['person']).shape[0]

    if debug:
        print('n_persons_transcript:', n_persons_transcript)

    # separate into two different DF to handle better time splits
    t_offers = transcript[transcript.event.str.contains('offer')]
    t_purchases = transcript[transcript.event.str.contains('transaction')]

    return (t_offers, t_purchases)


def load_profile(debug=False, location='colab'):
    """
    loads and processes the customer profile db

    :param debug: print debug information
    :param location: portfolio.json location (Google Colab or local)
    :return:
    """

    if location == 'colab':
        profile = pd.read_json(r'/content/gdrive/My Drive/UD/Sbux/data/profile.json', orient='records', lines=True)
    else:
        profile = pd.read_json(r'profile.json', orient='records', lines=True)

    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'].astype(str), format='%Y%m%d').astype(
        "datetime64")
    profile['joined_year'] = profile['became_member_on'].dt.year
    profile['joined_month'] = profile['became_member_on'].dt.month
    profile['joined_week'] = profile['became_member_on'].dt.week

#     profile['gender_num'] = profile.gender.astype("category", categories=['F', 'M', 'O']).cat.codes
    profile['gender_num'] = profile.gender.astype(pd.Categorical, categories=['F', 'M', 'O']).cat.codes

    time_start = profile.became_member_on.min()
    time_end = profile.became_member_on.max()
    time_delta = time_end - time_start

    if debug:
        print()
        print('time span in profile DB:', time_start, time_end)
        print('time_delta in profile DB:', time_delta.days, ' days ~ %d years' % np.round(time_delta.days / 365))

    profile['user_time'] = (time_end - profile['became_member_on']).dt.days
    profile = profile.rename(columns={'id': 'person'})

    return (profile)


def create_person_offer_table(transcript_offers, transcript_transactions, portfolio, add_missing=False, debug=False):
    """
    combines the transcript_offers, transcript_transactions, portfolio DBs into a single combined one that can be
    indexed by (person, offer, time)

    :param transcript_offers:
    :param transcript_transactions:
    :param portfolio:
    :param add_missing: adds back the persons (6) that did not receive an offer in the timeframe of transcript_offers
    :param debug:
    :return: s
    """
    # generates a main pivot table with customer and offer information (received, viewed, completed)

    # time of offer received / viewed / completed / reward
    combined_pot = extract_offer_times(transcript_offers, portfolio, debug=True);

    # adding the persons that don't receive any offer
    missing_persons = set(transcript_transactions.person) - set(combined_pot.person)
    print('persons with no offers received in timeframe: ', len(missing_persons))

    if add_missing:
        # adding back the persons that did not receive an offer in the timeframe of transcript_offers

        tmp1 = np.array(list(missing_persons))
        tmp = np.vstack((tmp1, np.empty(tmp1.shape) * np.nan)).T

        piv_missing = pd.DataFrame(tmp, columns=['person', 'offer'])
        # display(piv_missing)

        combined = pd.concat((combined_pot, piv_missing), sort=False)
    else:
        combined = combined_pot

    return combined


def add_offers_df(combined, portfolio, debug=False):
    """
    adds the offer details to the main pivot table

    :param combined:
    :param portfolio: portfolio of offers
    :param debug:
    :return:
    """

    #

    # merge the offers on (person, offer)
    if debug:
        print('combined.shape:', combined.shape)
    combined = combined.merge(portfolio[['difficulty', 'duration', 'offer', 'offer_type', 'web',
                                         'email', 'mobile', 'social', 'offer_reward', 'duration_hours']], on='offer',
                              how='left')
    combined.rename(columns={'offer completed': 'offer_completed', 'offer received': 'offer_received',
                             'offer viewed': 'offer_viewed'}, inplace=True)

    combined['offer_end'] = combined['offer_received'] + combined['duration_hours']
    combined['duration_viewed'] = combined['offer_end'] - combined['offer_viewed']
    combined['duration_viewed'] = combined['duration_viewed'].fillna(0)

    combined['duration_effective'] = combined[['duration_viewed', 'duration_hours']].min(axis=1, skipna=False)

    combined['viewed'] = np.invert(combined['offer_viewed'].isnull())
    combined['rewarded'] = np.invert(combined['reward'].isnull())
    combined['received'] = np.invert(combined['offer_received'].isnull())

    combined['comp_not_viewed'] = ((-combined['viewed']) & combined['rewarded'])
    combined['completed'] = combined['rewarded'] & (-combined['comp_not_viewed'])

    if debug:
        print('combined.shape:', combined.shape)

    return (combined)


def extract_offer_times(transcript_offers, portfolio, debug=True, ov_same_keep_earliest_viewed=True):
    """
    extracts the offer received, viewed, completed information from the transcript dataframe

    :param transcript_offers:
    :param portfolio:
    :param debug:
    :param ov_same_keep_earliest_viewed:
    :return: combined database by person, offer --> times (received, viewed, completed,...)
    """

    if debug:
        events_type = transcript_offers.event.value_counts()
        print(events_type)

    er = transcript_offers.loc[transcript_offers.event == 'offer received', ['person', 'offer', 'time']].rename(
        columns={'time': 'offer_received'})
    ec = transcript_offers.loc[
        transcript_offers.event == 'offer completed', ['person', 'offer', 'time', 'reward']].rename(
        columns={'time': 'offer_completed'})
    ev = transcript_offers.loc[transcript_offers.event == 'offer viewed', ['person', 'offer', 'time']].rename(
        columns={'time': 'offer_viewed'})

    if debug:
        print('er,ec,ev')
        print(er.shape, ec.shape, ev.shape)

    er = er.merge(portfolio[['offer', 'duration_hours']], on='offer')
    er['offer_end'] = er['offer_received'] + er['duration_hours'] - 1

    er.index.name = 'idx'
    er = er.reset_index()

    if debug:
        print('er ', er.shape)
    conn = sqlite3.connect(':memory:')

    er.to_sql('er', conn, index=False, if_exists='replace')  # ,index_label='idx')
    ec.to_sql('ec', conn, index=False, if_exists='replace')
    ev.to_sql('ev', conn, index=False, if_exists='replace')

    # SQL query for speed
    qry = '''
      SELECT  
          er.person, er.offer, er.offer_received, ev.offer_viewed, ec.offer_completed, ec.reward, er.idx, er.offer_end
      FROM
          er 
          left join  ec on 
          er.person = ec.person and er.offer = ec.offer and ec.offer_completed <= er.offer_end and ec.offer_completed >= er.offer_received
    
          left join  ev on 
          er.person = ev.person and er.offer = ev.offer and ev.offer_viewed <= er.offer_end and ev.offer_viewed >= er.offer_received   
    
      '''
    combined = pd.read_sql_query(qry, conn)  # .drop(columns=['idx'])

    # remove duplicates from SQL
    combined.drop_duplicates(subset=['idx'], inplace=True)
    combined = combined.drop(columns=['idx'])

    #   # dealing with overlapping offers of the same type

    #   if ov_same_keep_earliest_viewed != 'skip':
    #     piv['distance']=piv['offer_viewed']-piv['offer_received']

    #     ordered=piv.loc[:,['person','offer','offer_viewed','distance']].sort_values(by='distance',ascending=True)
    #     ordered=ordered.drop(columns=['distance'])

    #     if ov_same_keep_earliest_viewed:
    #       keep='first'
    #     else:
    #       keep='last'
    #     duplicated=ordered.loc[ordered.duplicated(keep=keep)].index

    #     piv=piv.drop(index=duplicated)
    #     piv=piv.drop(columns=['distance'])

    #     piv=piv.reindex()

    #   if debug:
    #     print(piv.shape)
    return (combined)



def extract_transactions(combined, transcript, add_transactions=False, skip_overlap=True, debug=False):
    """
    extract_transactions
    separates purchases from transcript_offers in those during/outside offers

    :param combined:
    :param transcript:
    :param add_transactions:
    :param skip_overlap: if offers are overlapping the transaction is assigned to the first offer received
    :return:
    """

    # filter the transactions for purchases
    trans = transcript[transcript['event'] == 'transaction'][['person', 'time', 'transaction']].sort_values(
        by=['person', 'time'])
    trans.rename(columns={'transaction': 'payments'}, inplace=True)
    trans.reset_index(inplace=True)

    if debug:
        print('trans.shape:', trans.shape)

    # extracts the transactions during an offer period, uses group by to avoid repeated rows in the SQL query
    conn = sqlite3.connect(':memory:')

    combined.to_sql('piv', conn, index=False, if_exists='replace')

    # (tr_id) index is the real index in the transcript DataFrame, idx is the reindexed index
    trans = trans.rename(columns={'index': 'tr_id'})

    # to check for duplicate lines after SQL query
    trans.index.name = 'idx'
    trans.reset_index().to_sql('trans', conn, index=False, if_exists='replace', index_label='idx')

    if debug:
        print(trans.columns)

    # https://stackoverflow.com/questions/30627968/merge-pandas-dataframes-where-one-value-is-between-two-others
    if skip_overlap:
        qry = '''
        SELECT  
            trans.person, trans.time, trans.payments, trans.tr_id, piv.offer, idx  
        FROM
            trans inner join  piv on 
            trans.person = piv.person and trans.time between offer_viewed and offer_end+1 
        WHERE
            viewed = 1
        GROUP BY
            idx
        '''
    else:
        qry = '''
    SELECT  
        trans.person, trans.time, trans.payments, trans.tr_id, piv.offer, idx , piv.offer_viewed
    FROM
        trans inner join  piv on 
        trans.person = piv.person and trans.time between offer_viewed and offer_end+1 
    WHERE
        viewed = 1
    '''

    transactions_during_offer = pd.read_sql_query(qry, conn)

    # check for duplicates and remove them. there should be none
    print('transactions_during_offer.shape:', transactions_during_offer.shape)

    if skip_overlap:
        # each transaction is attributed to the first offer received
        transactions_during_offer.set_index('idx', inplace=True)
        transactions_during_offer = transactions_during_offer.groupby(by='idx').first()
    else:
        # fix the issue that some transactions are counted multiple times on the same offer
        transactions_during_offer = transactions_during_offer.drop_duplicates(
            subset=['person', 'time', 'payments', 'tr_id', 'offer'])

        # each transaction is attributed to all the valid offers
        vc = transactions_during_offer['tr_id'].value_counts().rename('overlaps')
        transactions_during_offer = transactions_during_offer.merge(vc, left_on='tr_id', right_index=True)
        # overlaps = value_counts -1
        transactions_during_offer['overlaps'] -= 1

    if debug:
        print('combined.shape:', combined.shape)
        print('trans.shape:', trans.shape)

        print('transactions_during_offer.shape:', transactions_during_offer.shape)
        print('transactions_during_offer index unique :', len(set(transactions_during_offer.index)))
    n_trans_do = transactions_during_offer.shape[0]

    if add_transactions:
        combined = combined.merge(transactions_during_offer, on=['person', 'offer'], how='left')
        print('piv.shape:', combined.shape)

    #  get the remaining transactions
    ind = trans.index.difference(transactions_during_offer.index)
    transactions_outside_offer = trans.loc[ind, :]

    # check validity
    if debug:
        print()
        #   print('trans.shape:',trans.shape)
        print('transactions_outside_offer.shape:', transactions_outside_offer.shape)
        print('transactions_during_offer.shape:', transactions_during_offer.shape)
        print('total transactions: ', transactions_outside_offer.shape[0] + transactions_during_offer.shape[0])
        print('difference: ', transactions_outside_offer.shape[0] + transactions_during_offer.shape[0] - trans.shape[0])

    assert trans.shape[0] == transactions_outside_offer.shape[0] + transactions_during_offer.shape[
        0], "transactions not matching "

    # trans is the full transaction database
    return (combined, trans, transactions_during_offer, transactions_outside_offer)


def assign_overlapping_purchases(transactions_during_offer, assign_to='first'):
    """
    assign_overlapping_purchases
    resolve case of overlapping offers by the criteria assign_to

    :param transactions_during_offer:
    :param assign_to: first, last, ignore
    :return:
    """

    # resolve case of overlapping offers by
    conn = sqlite3.connect(':memory:')

    # find purchases with overlapping offers
    tdo_ov = transactions_during_offer.loc[(transactions_during_offer.overlaps > 0)]

    tdo_ov.to_sql('tdo', conn, index=False, if_exists='replace')

    # assign to first viewed offer
    if assign_to == 'first':
        qry = '''
    SELECT  
         person,  time,  payments, tr_id,  offer,   idx, MIN(offer_viewed) AS offer_viewed
    FROM
        tdo
    GROUP BY 
        tr_id
    '''

    # assign to last viewed offer
    elif assign_to == 'last':
        qry = '''
    SELECT  
         person,  time,  payments, tr_id,  offer,   idx, MAX(offer_viewed) AS offer_viewed
    FROM
        tdo
    GROUP BY 
        tr_id
    '''
    elif assign_to == 'ignore':
        return transactions_during_offer
    else:
        print('method not selected!')
        return (None)

    tdo_nov = pd.read_sql_query(qry, conn)
    if len(tdo_nov) > 0:
        tdo_nov['overlaps'] = 0
    else:
        print('tdo_nov:', tdo_nov)

    tdo_others = transactions_during_offer.loc[(transactions_during_offer.overlaps == 0)]

    return pd.concat([tdo_nov, tdo_others], ignore_index=True)


def add_stats_by_person(combined, transactions_outside_offer, transactions_during_offer, calc_net_offer_time=False):
    """
    add_stats_by_person

    :param combined:
    :param transactions_outside_offer:
    :param transactions_during_offer:
    :param calc_net_offer_time:
    :return:
    """

    # customers who made purchases but in times when no offer is received  (total $ sum and number of purchases)
    total_purchase_outside = transactions_outside_offer.groupby(by=['person'])['payments'].agg('sum').to_frame()
    total_purchase_outside.rename(columns={'payments': 'Tpay_out'}, inplace=True)

    total_purchase_outside_num = transactions_outside_offer.groupby(by=['person'])['payments'].agg('count').to_frame()
    total_purchase_outside_num.rename(columns={'payments': 'Npay_out'}, inplace=True)

    t_offers = transactions_during_offer.groupby(by=['person'])['payments'].agg(['sum', 'count', 'max', 'min'])
    t_offers.rename(columns={'sum': 'Tpay_offers_tot', 'count': 'Npay_offers_tot', 'max': 'Maxpay_offers_tot',
                             'min': 'Minpay_offers_tot'}, inplace=True)
    t_offers = t_offers.fillna(0)

    t_rewards = combined.groupby(by=['person'])['reward'].agg(['sum', 'count'])
    t_rewards.rename(columns={'sum': 'Trewards_tot', 'count': 'Nrewards_tot'}, inplace=True)
    t_rewards = t_rewards.fillna(0)

    ## SPECIFIC FOR A CUSTOMER
    # total offers time (gross, ie counting multiple times overlapping time intervals only once)
    offer_times = combined.groupby(by='person')['duration_effective'].agg('sum').to_frame()
    offer_times.rename(columns={'duration_effective': 'tot_gross_offer_time'}, inplace=True)

    # total offers time (gross, ie counting multiple times overlapping offer intervals)
    offer_timesN = combined.groupby(by=['person', 'offer'])[['offer_viewed', 'offer_end']].agg(np.mean)
    offer_timesN = offer_timesN[pd.notnull(offer_timesN['offer_viewed'])]

    if calc_net_offer_time:
        tot_net_offers_time = overlapping_offer_intevals(combined, final_time=744, return_person_occupacy=False)

    # completed offers
    p_offers_completed = combined.groupby(by='person')['completed', 'comp_not_viewed', 'received','viewed'].agg('sum')
    p_offers_completed['p_c_r_ratio'] = p_offers_completed['completed'] / p_offers_completed['received']
    p_offers_completed['p_c_v_ratio'] = p_offers_completed['completed'] / p_offers_completed['viewed']

    p_offers_completed['p_cnv_r_ratio'] = p_offers_completed['comp_not_viewed'] / p_offers_completed['received']
    p_offers_completed=p_offers_completed.fillna(0)

    print('add_stats_by_person:')
    print(combined.shape)
    #   print('transactions_during_offer',transactions_during_offer.shape)
    # add overlaps
    combined_with_person_stats = combined.merge(total_purchase_outside, on=['person'], how='left')

    combined_with_person_stats = combined_with_person_stats.merge(total_purchase_outside_num, on=['person'], how='left')
    combined_with_person_stats = combined_with_person_stats.merge(p_offers_completed[['p_c_r_ratio', 'p_cnv_r_ratio','p_c_v_ratio']],
                                                                  on=['person'], how='left')

    overlaps = transactions_during_offer[['person', 'offer', 'offer_viewed', 'overlaps']].groupby(
        by=['person', 'offer', 'offer_viewed']).agg('sum')
    combined_with_person_stats = combined_with_person_stats.merge(overlaps, on=['person', 'offer', 'offer_viewed'],
                                                                  how='left')
    #   piv2=piv2.merge( transactions_during_offer[['person', 'offer',  'offer_viewed', 'overlaps']].drop_duplicates() ,on=['person','offer','offer_viewed'],how='left')

    print(combined_with_person_stats.shape)

    combined_with_person_stats = combined_with_person_stats.merge(t_offers, on=['person'], how='left')
    combined_with_person_stats = combined_with_person_stats.merge(t_rewards, on=['person'], how='left')
    #   piv2=piv2.merge(t_offers_num,on=['person'],how='left')
    print(combined_with_person_stats.shape)

    combined_with_person_stats = combined_with_person_stats.merge(offer_times, on='person', how='left')
    print(combined_with_person_stats.shape)

    # piv2[['tot_net_offers_time'],'tot_net_offers_time','reward','Tpay_offer','Tpay_out','Npay_out','Npay_offer'
    if calc_net_offer_time:
        combined_with_person_stats = combined_with_person_stats.merge(tot_net_offers_time.reset_index(), on=['person'],
                                                                      how='left')
        combined_with_person_stats['tot_net_offers_time'] = combined_with_person_stats['tot_net_offers_time'].fillna(0)
        combined_with_person_stats['tot_not_offers_time'] = 744. - combined_with_person_stats['tot_net_offers_time']

        combined_with_person_stats['Avg_pay_offers'] = combined_with_person_stats['Tpay_offers_tot'] / \
                                                       combined_with_person_stats['tot_net_offers_time']
        combined_with_person_stats['Net_pay_offers'] = combined_with_person_stats['Tpay_offers_tot'] - \
                                                       combined_with_person_stats['Trewards_tot']
        combined_with_person_stats['Avg_net_pay_offers'] = (combined_with_person_stats['Tpay_offers_tot'] -
                                                            combined_with_person_stats['Trewards_tot']) / \
                                                           combined_with_person_stats['tot_net_offers_time']

        combined_with_person_stats['Avg_pay_outside'] = combined_with_person_stats['Tpay_out'] / \
                                                        combined_with_person_stats['tot_not_offers_time']
        combined_with_person_stats['Avg_pay_offers'] = combined_with_person_stats['Avg_pay_offers'].fillna(0)
        combined_with_person_stats['Avg_pay_outside'] = combined_with_person_stats['Avg_pay_outside'].fillna(0)

        combined_with_person_stats['Avg_D_O'] = combined_with_person_stats['Avg_pay_offer'] - \
                                                combined_with_person_stats['Avg_pay_outside']
        combined_with_person_stats['Avg_D_OS'] = combined_with_person_stats['Avg_pay_offers'] - \
                                                 combined_with_person_stats['Avg_pay_outside']
        combined_with_person_stats['Avg_D_nOS'] = combined_with_person_stats['Avg_net_pay_offers'] - \
                                                 combined_with_person_stats['Avg_pay_outside']

        combined_with_person_stats['Avg_D_nOS']=combined_with_person_stats['Avg_D_nOS'].fillna(0)


    combined_with_person_stats['reward'] = combined_with_person_stats['reward'].fillna(0)
    combined_with_person_stats['Tpay_offers_tot'] = combined_with_person_stats['Tpay_offers_tot'].fillna(0)
    combined_with_person_stats['Tpay_out'] = combined_with_person_stats['Tpay_out'].fillna(0)
    combined_with_person_stats['Npay_out'] = combined_with_person_stats['Npay_out'].fillna(0)
    combined_with_person_stats['Npay_offers_tot'] = combined_with_person_stats['Npay_offers_tot'].fillna(0)
    combined_with_person_stats['Avg_net_pay_offers'] = combined_with_person_stats['Avg_net_pay_offers'].fillna(0)

    combined_with_person_stats['Maxpay_offer'] = combined_with_person_stats['Maxpay_offer'].fillna(0)
    combined_with_person_stats['Minpay_offer'] = combined_with_person_stats['Minpay_offer'].fillna(0)
    combined_with_person_stats['Netpay_offer'] = combined_with_person_stats['Netpay_offer'].fillna(0)
    combined_with_person_stats['Maxpay_offers_tot'] = combined_with_person_stats['Maxpay_offers_tot'].fillna(0)
    combined_with_person_stats['Minpay_offers_tot'] = combined_with_person_stats['Minpay_offers_tot'].fillna(0)
    combined_with_person_stats['Net_pay_offers'] = combined_with_person_stats['Net_pay_offers'].fillna(0)
    combined_with_person_stats['overlaps'] = combined_with_person_stats['overlaps'].fillna(0)


    print(combined_with_person_stats.shape)

    return (combined_with_person_stats)


def add_purchases(combined, profile, transactions_outside_offer, tdo, assign_to='first', calc_net_offer_time=False,
                  add_person_stats=True, debug=False):
    """
    adds the purchase information (during and outside offers) to the main pivot table.
    the information of average spending during and outside of offers is calculated

    :param combined:
    :param profile:
    :param transactions_outside_offer:
    :param tdo:
    :param assign_to:
    :param calc_net_offer_time:
    :param add_person_stats:
    :return:
    """

    # deal with multiple offers
    transactions_during_offer = assign_overlapping_purchases(tdo, assign_to=assign_to)

    # this should be done in a separate function
    transactions_during_offer['payments_original'] = 1.0 * transactions_during_offer['payments']
    if assign_to == 'ignore':
        # a payment is received during multiple offers are received, the amount is split equally between the offers
        transactions_during_offer['payments'] = transactions_during_offer['payments_original'] / (
                transactions_during_offer['overlaps'] + 1.)

    t_spec_offer = transactions_during_offer.groupby(by=['person', 'offer', 'offer_viewed'])['payments'].agg(
        ['sum', 'count', 'max', 'min'])
    t_spec_offer.rename(
        columns={'sum': 'Tpay_offer', 'count': 'Npay_offer', 'max': 'Maxpay_offer', 'min': 'Minpay_offer', },
        inplace=True)

    # adding the total purchases outside of offers and during offer (specific and total)
    if debug:
        print('add_purchases:')
        print(combined.shape)

    combined_with_purchases = combined.merge(t_spec_offer, on=['person', 'offer', 'offer_viewed'], how='left')
    print(combined_with_purchases.shape)

    combined_with_purchases['Tpay_offer'] = combined_with_purchases['Tpay_offer'].fillna(0)
    combined_with_purchases['Npay_offer'] = combined_with_purchases['Npay_offer'].fillna(0)

    combined_with_purchases['Netpay_offer'] = combined_with_purchases['Tpay_offer'] - combined_with_purchases['reward']

    # if the offer is not seen, it should not be considered. It also prevents the division by zero
    # prevents division by zero if there's a payment
    combined_with_purchases['Avg_pay_offer'] = combined_with_purchases['Tpay_offer'] / np.maximum(
        combined_with_purchases['duration_effective'],
        1.0 + 0.0 * combined_with_purchases['duration_effective'])
    combined_with_purchases.loc[(combined_with_purchases['duration_effective'] == 0), 'Avg_pay_offer'] = 0.0
    combined_with_purchases['Avg_pay_offer'] = combined_with_purchases['Avg_pay_offer'].fillna(0)

    combined_with_purchases = combined_with_purchases.merge(profile, on='person', how='left')

    if add_person_stats:
        combined_with_purchases = add_stats_by_person(combined_with_purchases, transactions_outside_offer,
                                                      transactions_during_offer,
                                                      calc_net_offer_time=calc_net_offer_time)

    return (combined_with_purchases)


def create_pivot_table(portfolio, profile, transcript_offers, transcript_transactions, add_transactions=False,
                       skip_overlap=False,
                       skip_add_purchases=False, calc_net_offer_time=True, assign_to='first'):
    """
    creates the pivot table with all the derived information

    :param portfolio:
    :param profile:
    :param transcript_offers:
    :param transcript_transactions:
    :param add_transactions:
    :param skip_overlap:
    :param skip_add_purchases:
    :param calc_net_offer_time:
    :param assign_to:
    :return:
    """

    combined = create_person_offer_table(transcript_offers, transcript_transactions, portfolio)
    combined = add_offers_df(combined, portfolio);
    combined, trans, transactions_during_offer, transactions_outside_offer = extract_transactions(combined,
                                                                                                  transcript_transactions,
                                                                                                  add_transactions=add_transactions,
                                                                                                  skip_overlap=skip_overlap);
    if not skip_add_purchases:
        combined = add_purchases(combined, profile, transactions_outside_offer, transactions_during_offer,
                                 calc_net_offer_time=calc_net_offer_time, assign_to=assign_to)

    return (combined, transactions_during_offer, transactions_outside_offer, trans)


def load_data_cv(person_split=None, rename_offers=False, time_split_min=None, time_split_max=None,
                 add_transactions=False, skip_overlap=False,
                 skip_add_purchases=False, calc_net_offer_time=False, assign_to='first',location='colab'):
    """
    creates the main table and transaction tables
    can also generate CV splits by persons (by fraction) or by time (before a time in hours)

    :param person_split:
    :param rename_offers:
    :param time_split_min:
    :param time_split_max:
    :param add_transactions:
    :param skip_overlap:
    :param skip_add_purchases:
    :param calc_net_offer_time:
    :param assign_to:
    :return:
    """

    # print(person_split)

    rename_offers_dict = {'0b1e1539f2cc45b7b9fa7c272da2e1d7': 'a',
                          '2298d6c36e964ae4a3e7e9706d1fb8c2': 'b',
                          '2906b810c7d4411798c6938adc9daaa5': 'c',
                          '3f207df678b143eea3cee63160fa8bed': 'd',
                          '4d5c57ea9a6940dd891ad53e9dbe8da0': 'e',
                          '5a8bc65990b245e5a138643cd4eb9837': 'f',
                          '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'g',
                          'ae264e3637204a6fb9bb56bc8210ddfd': 'h',
                          'f19421c1d4aa40978ebb69ca19b0e20d': 'i',
                          'fafdcd668e3743c1bb461111dcafc2a4': 'j'}

    if ((not time_split_min is None) & (not person_split is None)):
        print('CV split must be either by person or by time')
        sys.exit(1)

    portfolio = load_portfolio(location=location)
    offers_all, transactions_all = load_transcript(location=location)
    profile_all = load_profile(location=location)

    if rename_offers:
        portfolio['offer'] = portfolio['offer'].replace(rename_offers_dict)
        offers_all['offer'] = offers_all['offer'].replace(rename_offers_dict)
        transactions_all['offer'] = transactions_all['offer'].replace(rename_offers_dict)

    if ((person_split is None) & (time_split_min is None) & (time_split_max is None)):
        combined, transactions_during_offer, transactions_outside_offer, trans = create_pivot_table(portfolio,
                                                                                                    profile_all,
                                                                                                    offers_all,
                                                                                                    transactions_all,
                                                                                                    add_transactions=add_transactions,
                                                                                                    skip_overlap=skip_overlap,
                                                                                                    skip_add_purchases=skip_add_purchases,
                                                                                                    calc_net_offer_time=calc_net_offer_time,
                                                                                                    assign_to=assign_to)

        return (
        combined, profile_all, portfolio, offers_all, trans, transactions_during_offer, transactions_outside_offer)

        # creating CV splitting by persons:
    print()
    if person_split is None:
        print('using full person list')

        if not time_split_min is None:
            print()
            print()
            print('min time split:')
            print("time: ", offers_all.time.min(), offers_all.time.max())

            offers = offers_all.loc[offers_all.time < time_split_min, :]
            transactions = transactions_all.loc[transactions_all.time < time_split_min, :]

            if time_split_max is None:
                print('time splitting without max')
                offers_test = offers_all.loc[offers_all.time >= time_split_min, :]
                transactions_test = transactions_all.loc[transactions_all.time >= time_split_min, :]

            else:
                print('time splitting with max')
                offers_test = offers_all.loc[offers_all.time >= time_split_max, :]
                transactions_test = transactions_all.loc[transactions_all.time >= time_split_max, :]
                print("offers.shape: ", offers.shape)
                print("offers_test.shape: ", offers_test.shape)

    else:
        # person split
        Lprofile, Lprofile_test = train_test_split(profile_all.person, test_size=person_split)
        print('train set %d persons' % Lprofile.shape[0])

        profile = profile_all.loc[profile_all.person.isin(list(Lprofile))]
        profile_test = profile_all.loc[profile_all.person.isin(list(Lprofile_test))]

        offers = offers_all.loc[offers_all.person.isin(list(Lprofile))]
        offers_test = offers_all.loc[offers_all.person.isin(list(Lprofile_test))]

        transactions = transactions_all.loc[transactions_all.person.isin(list(Lprofile))]
        transactions_test = transactions_all.loc[transactions_all.person.isin(list(Lprofile_test))]

    combined, transactions_during_offer, transactions_outside_offer, trans = create_pivot_table(portfolio, profile_all,
                                                                                                offers, transactions,
                                                                                                add_transactions=add_transactions,
                                                                                                skip_overlap=skip_overlap,
                                                                                                skip_add_purchases=skip_add_purchases,
                                                                                                calc_net_offer_time=calc_net_offer_time,
                                                                                                assign_to=assign_to)
    out = {}
    out['train'] = combined
    out['train_tdo'] = transactions_during_offer
    out['train_too'] = transactions_outside_offer
    out['train_offers'] = offers
    out['train_transactions'] = trans

    if person_split is None:
        if not time_split_min is None:
            #       return(out ,profile_all,portfolio,offers_all,transactions_all )
            #     else:
            # time split
            combined_test, transactions_during_offer_test, transactions_outside_offer_test, trans_test = create_pivot_table(
                portfolio, profile_all, offers_test, transactions_test,
                add_transactions=add_transactions, skip_overlap=skip_overlap,
                skip_add_purchases=skip_add_purchases,
                calc_net_offer_time=calc_net_offer_time, assign_to=assign_to)
            out['test'] = combined_test
            out['test_tdo'] = transactions_during_offer_test
            out['test_too'] = transactions_outside_offer_test
            out['test_offers'] = offers_test
            out['test_transactions'] = trans_test

            return (out, profile_all, portfolio, offers_all, transactions_all)

    else:
        # person split
        print('test data')
        combined_test, transactions_during_offer_test, transactions_outside_offer_test, trans_test = create_pivot_table(
            portfolio, profile_test, offers_test, transactions_test,
            add_transactions=add_transactions, skip_overlap=skip_overlap,
            skip_add_purchases=skip_add_purchases,
            calc_net_offer_time=calc_net_offer_time, assign_to=assign_to)
        out['test'] = combined_test
        out['test_tdo'] = transactions_during_offer_test
        out['test_too'] = transactions_outside_offer_test
        out['test_offers'] = offers_test
        out['test_transactions'] = trans_test

        return (out, profile_all, portfolio, offers_all, transactions_all)


def overlapping_offer_intevals(pot_df, final_time=744, return_person_occupacy=True):
    """

    :param pot_df:
    :param final_time:
    :param return_person_occupacy:
    :return:
    """


    # derive the net offer time for a person, counting overlapping offers only once

    # work on subset pot (person, offer, time)
    pot = pot_df[['person', 'offer', 'offer_viewed', 'offer_end']].copy().dropna(subset=['offer_viewed'])
    pot['offer_viewed'] = pot['offer_viewed'].astype(int)
    pot['offer_end'] = pot['offer_end'].astype(int)

    start_time = pot['offer_viewed'].values
    end_time = np.minimum(pot['offer_end'].values, final_time + 0 * pot['offer_end'].values)
    ind = np.arange(0, len(pot))

    # Repeat start position index length times and concatenate
    lengths = end_time - start_time + 1
    cat_start = np.repeat(start_time, lengths)
    y_range = np.repeat(ind, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)

    # Add group counter to group specific starts
    x_range = cat_start + cat_counter

    # sparse matrix to vectorize computation
    occupancy = coo_matrix((1 + 0 * x_range, (y_range, x_range)))

    occupancy_df = pd.DataFrame(occupancy.toarray(), index=pot.index, columns=range(0, final_time + 1))

    person_occupacy = pd.concat((pot.drop(columns=['offer_viewed', 'offer_end']), occupancy_df), axis=1)

    # aggregate by person
    person_occupacy = person_occupacy.groupby(by='person').agg('sum')
    person_occupacy = person_occupacy.clip(upper=1)

    effective_duration = person_occupacy.sum(axis=1).to_frame().rename(columns={0: 'tot_net_offers_time'})
    if return_person_occupacy:
        return (effective_duration, person_occupacy)
    else:
        return (effective_duration)


def plot_offers_of_customer_time(piv, trans, tdo, too, customer, yc=0, step=0.1, plot_transactions=True,
                                 plot_offer_name=True, overlay_spacing=0.2):
    """
    plots the offers and the transactions for a customer versus time

    :param piv:
    :param trans:
    :param tdo:
    :param too:
    :param customer:
    :param yc:
    :param step:
    :param plot_transactions:
    :param plot_offer_name:
    :param overlay_spacing:
    :return:
    """

    gg = piv.loc[(piv.person == customer), 'offer_received'].drop_duplicates()
    ind = gg.index.values

    viewed_offers = []

    v_off_times = np.zeros((ind.shape[0], 2))
    v_off_set = set([])

    c = 0

    offer_colors = []

    for n in ind:
        hh = plt.plot([piv.loc[n, 'offer_received'], piv.loc[n, 'offer_end']], [yc, yc], 'o-', alpha=0.6)
        color = hh[0].get_color()
        offer_colors.append(color)

        if plot_offer_name:
            plt.text((piv.loc[n, 'offer_received'] + piv.loc[n, 'offer_end']) / 2, yc - .2, piv.loc[n, 'offer']);

        if not np.isnan(piv.loc[n, 'offer_viewed']):
            plt.plot([piv.loc[n, 'offer_viewed'], piv.loc[n, 'offer_end']], [yc, yc], '.-', alpha=1.0, color=color)
            viewed_offers.append(piv.loc[n, 'offer'])
            v_off_times[c, 0] = piv.loc[n, 'offer_viewed']
            v_off_times[c, 1] = piv.loc[n, 'offer_end']
            v_off_set = v_off_set.union(set(range(int(piv.loc[n, 'offer_viewed']), int(piv.loc[n, 'offer_end']) + 0)))
            c += 1

        yc += step
    v_off_times = v_off_times[0:c, :]

    diff = np.diff(np.array(list(v_off_set)))
    diff[diff > 1] = 0
    overlapping_offers_time = diff.sum()

    if plot_transactions:
        df = trans.loc[(trans.person == customer)]
        plt.plot(df['time'], df['payments'], 'o', color='k', alpha=0.3)

        # show transactions attributed to several offers
        n_offer = 0

        for n in ind:
            y_offer = n_offer + 1
            tdoc = tdo.loc[(tdo.person == customer) & (tdo.offer == piv.loc[n, 'offer']) & (
                    tdo.offer_viewed == piv.loc[n, 'offer_viewed'])]
            dy = (tdoc['overlaps']) * y_offer
            if (tdoc['overlaps'] > 0).any():
                y_offer += overlay_spacing

            if n_offer < len(offer_colors):
                #         print(n_offer)

                #         print(offer_colors[n_offer])
                plt.plot(tdoc['time'], tdoc['payments'] + dy, '.', color=offer_colors[n_offer], alpha=0.8)
            else:
                plt.plot(tdoc['time'], tdoc['payments'] + dy, '.', alpha=0.8)
            n_offer += 1

    plt.xlabel('hours');
    plt.ylabel('purchases');

    return (viewed_offers, v_off_times, overlapping_offers_time)


if __name__ == "__main__":
    out, profile_all, portfolio, offers_all, transactions_all, transactions_during_offer, transactions_outside_offer = load_data_cv(
        person_split=None, rename_offers=True, time_split_min=None, time_split_max=None,
        add_transactions=False, skip_overlap=False, skip_add_purchases=False, calc_net_offer_time=True,
        assign_to='ignore', location='local')