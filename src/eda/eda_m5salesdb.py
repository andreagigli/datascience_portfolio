from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def eda(sales):
    """
    Performs exploratory data analysis (EDA) on one or more datasets provided as arguments.

    Args:
        *args: Variable length argument list. Each argument can be a Pandas DataFrame, a series, or any
               data structure suitable for analysis. The function can also accept arguments specifying
               details about the analysis to be performed, such as column names of interest, types of
               plots to generate, or specific statistics to compute.

    Returns:
        The function's return value would typically include summaries of the analyses performed, such as
        printed output of statistical tests, matplotlib figures or seaborn plots visualizing aspects of the
        data, or even a list or dictionary summarizing key findings. The specific return type and structure
        would depend on the implementation of the EDA tasks within the function.
    """

    """
    The sales dataset comprises: 
    * the target variable (no. sold items)
    * categorical aggregation levels (state, store, item category, item department)
    * the price of each item
    * misc information such as special events
    
    Some interesting eda questions may be: 
    * Characterize the number of items for wrt different aggregation levels. This means no items wrt state, no items
    wrt shop, no items wrt category, no items wrt shop and category
    * Characterizing the target variable wrt different aggregation layers. This means sales wrt state, sales wrt shop,
    sales wrt category, sales wrt shop and cathegory
    * Identify potential trends in the target variable. This means sales over time, sales over time and over a 
    periodic interval (heatmap)
    """

    print("\n\nCharacterize the number of unique items across different dimensions")
    items_per_state = sales.groupby('state_id')['item_id'].nunique()
    items_per_shop = sales.groupby('store_id')['item_id'].nunique()
    items_per_category = sales.groupby('cat_id')['item_id'].nunique()
    items_per_shop_and_category = sales.groupby(['state_id', 'store_id', 'cat_id'])['item_id'].nunique().reset_index(name='unique_items')
    # print("Number of unique items per state:\n", items_per_state)
    # print("\nNumber of unique items per shop (store):\n", items_per_shop)
    # print("\nNumber of unique items per category:\n", items_per_category)
    # print("\nNumber of unique items per shop and category combination:\n", items_per_shop_and_category)

    # # Create a bar plot showing the number of unique items for each shop and category combination.
    # plt.figure(figsize=(12, 8))
    # sns.barplot(x='store_id', y='unique_items', hue='cat_id', data=items_per_shop_and_category)
    # plt.title('Number of Unique Items per Shop and Category')
    # plt.xlabel('Store ID')
    # plt.ylabel('Number of Unique Items')
    # plt.xticks(rotation=45)
    # plt.legend(title='Category')
    # plt.tight_layout()

    # Create a sunburst chart using Plotly to visualize the hierarchical relationship of unique items across states,
    # stores, and categories. The 'path' defines the hierarchy levels, 'values' define the sizes of the sectors,
    # and 'color' reflects the count, with a gradient scale for visual distinction.
    items_per_shop_and_category['label'] = (items_per_shop_and_category['cat_id'] +
                                            ' (' + items_per_shop_and_category['unique_items'].astype(str) + ')')
    fig = px.sunburst(
        items_per_shop_and_category,
        path=['state_id', 'store_id', 'label'],
        values='unique_items',
        title='Sunburst Chart of Unique Items per Shop and Category',
        color='unique_items',
        color_continuous_scale='Blues'  # Optional: use a color scale to represent counts
    )
    fig.show()
    # fig.write_image("sunburst_chart.png")  # Save plotly figure as static image using kaleido package

    # Explore the total number of sold items per day per store
    daily_sales_per_store = sales.groupby(["state_id", "store_id", "date"])['sold'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=daily_sales_per_store, x='store_id', y='sold', hue="state_id")
    plt.title('Distribution of Total Sold Items per Day per Store')
    plt.xlabel('Shop')
    plt.ylabel('Total Sold Items')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # Explore the historical evolution of the daily revenue per store. The daily revenue is the daily_sum(item_cost*item_sold_pieces)
    daily_revenue_per_store = sales.groupby(['store_id', 'state_id', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(name='total_revenue')  # this ensures that the store_id, state_id and date are columns of the new df and the aggregated column is names total_revenue
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_revenue_per_store, x='date', y='total_revenue', hue="state_id", style="store_id")
    plt.title('Historical Evolution of Daily Revenue per Store')
    plt.xlabel('Shop')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # Explore the historical evolution of the daily revenue per state and item category
    daily_revenue_per_cat_per_store = sales.groupby(['store_id', 'state_id', 'cat_id', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(name='total_revenue')  # this reset ensures that the store_id, state_id and date are columns of the new df and the aggregated column is names total_revenue
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_revenue_per_cat_per_store, x='date', y='total_revenue', hue="state_id", style="cat_id")
    plt.title('Historical Evolution of Daily Revenue per State and Item Category')
    plt.xlabel('Shop')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # ... and the historical evoluation of daily revenue per state, store and item category
    g = sns.relplot(
        data=daily_revenue_per_cat_per_store,
        x='date', y='total_revenue',
        row='state_id',
        hue='store_id',  # Differentiates lines by store within each plot
        style='cat_id',  # Differentiates lines by category within each plot
        kind='line',  # Specifies that we want lineplots
        facet_kws={'sharey': False, 'sharex': True},
        height=3, aspect=2,  # Controls the size of each subplot
    )
    plt.suptitle("Historical Evolution of Daily Revenue per State, Store and Item Category")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    g.fig.tight_layout()
    g.fig.show(block=False)

    # Focusing on one specific store: historical evolution of daily revenue per item category, with indication of special promotions (SNAP)
    specific_store_revenue = daily_revenue_per_cat_per_store[daily_revenue_per_cat_per_store['store_id'] == "CA_1"]
    snap_dates = sales[sales['snap_CA'] == 1]['date'].drop_duplicates()
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(data=specific_store_revenue, x='date', y='total_revenue', hue='cat_id', style='cat_id')
    plt.title('Historical Evolution of Daily Revenue per Category for Store CA_1')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.legend(title='Item Category')
    for snap_date in snap_dates:
        ax.axvspan(snap_date, snap_date, color='grey', alpha=0.2)  # Adjust alpha for transparency
    plt.tight_layout()
    plt.show(block=False)

    # Focusing on one specific store: distribution of the daily revenue for food items (which are influenced by SNAP), in SNAP vs non-SNAP days
    daily_revenue_per_cat_per_store_snap = sales.groupby(['store_id', 'state_id', 'cat_id', 'snap_CA', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(name='total_revenue')
    specific_store_revenue_snap = daily_revenue_per_cat_per_store_snap[daily_revenue_per_cat_per_store['store_id'] == "CA_1"]
    plt.figure(figsize=(14, 7))
    ax = sns.violinplot(data=specific_store_revenue_snap, x='snap_CA', y='total_revenue', hue='cat_id')
    plt.title('Distribution of Daily Revenue per Item Category in SNAP vs. Non-SNAP Days for Store CA_1 ')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.xticks(ticks=[0, 1], labels=["Non-SNAP", "SNAP"], rotation=0)
    plt.legend(title='Item Category')
    plt.tight_layout()
    plt.show(block=False)

    # Focusing on one specific store: historical evolution of daily revenue detailed over two time scales (daily=weekday, and weekly=week_number)
    specific_store_revenue.loc[:, 'weekday'] = specific_store_revenue['date'].dt.day_name()
    specific_store_revenue.loc[:, 'week_number'] = specific_store_revenue['date'].dt.isocalendar().week
    # If you want to check if aggregation is necessary for the values of the pivot table, use
    # (specific_store_revenue.groupby(['weekday', 'week_number']).size() == 1).all(). In this case, aggregation is
    # necessary over the cat_id.
    pivot_table = specific_store_revenue.pivot_table(index='weekday', columns='week_number', values='total_revenue', aggfunc='sum')
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.loc[weekdays, :]  # This ensures the columns are in the correct order
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis', annot=False, fmt=".2f", linewidths=.5)
    plt.title('Weekly Revenue Distribution for Store CA_1')
    plt.xlabel('Day of the Week')
    plt.ylabel('Week Number of the Year')
    plt.tight_layout()
    plt.show(block=False)

    pass
    print(1)
    return