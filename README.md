# AFLW goalkicker model

A self-updating website with fair odds for anytime (1+), 2+ and 3+ goalkicker markets, an odds checker, and the model's track record.

## Put the website online (about 10 minutes, free)

1. **Make a GitHub account** at github.com/signup, if you don't have one.
2. **Create a repository.** Click **+** (top right), then **New repository**.
   - Name: `aflw-goalkickers` (or anything you like)
   - Choose **Public**. GitHub Pages is free for public repositories. Anyone could find the code and data, but nobody will unless you share the link.
   - Leave the other options as they are and click **Create repository**.
3. **Upload the files.** On the new repo page, click **uploading an existing file**. Open the unzipped `aflw_gk` folder, select **everything inside it** (including the `.github` folder) and drag it all onto the page. Click **Commit changes**.
   - Check that a `.github` folder now appears in the repo's file list. If it doesn't, click **Add file**, then **Create new file**, name it `.github/workflows/update.yml`, paste in the contents of that file from the zip, and commit.
4. **Turn on Pages.** Go to **Settings**, then **Pages**. Under *Build and deployment*, set **Source** to **GitHub Actions**.
5. **Run it for the first time.** Go to the **Actions** tab. If GitHub asks, click to enable workflows. Click **Update site**, then **Run workflow**, then **Run workflow** again. It takes about 3 minutes.
6. **Open the site** at `https://<your-username>.github.io/aflw-goalkickers/`, and bookmark it on your phone.

After that it updates every 3 hours on its own: results after each round, named teams once clubs announce them (usually Thursday), and late changes up to game day.

**If a run fails** (a red ✗ on the Actions tab), click into it and see which step failed:

- *Pull new AFLW data*: the AFL site may be blocking GitHub's servers. Send me the error.
- *Save new data to the repo*, with a permission error: go to **Settings**, then **Actions**, then **General**, set **Workflow permissions** to **Read and write**, and re-run.

GitHub pauses scheduled runs if a repo has no activity for 60 days, which will happen in the off-season. Before next season, go to the **Actions** tab and re-enable the workflow.

## Using the site

- **Fair odds:** pick a game, pick a market (1+, 2+ or 3+), and type the bookie's price next to a player. The site shows your edge (chance × odds − 1) and flags anything at **+10% or more**, with a stake of a quarter of the Kelly amount, capped at 2% of the bank you enter. The odds you type are saved only in your own browser.
- Each team shows **✓ Teams named** or **⏳ Last week's lineup**. Don't bet off last week's lineup.
- **Track record:** out-of-sample accuracy for the current season, a calibration chart, and how last round's top picks went.

## Running it on your PC instead (optional)

```
pip install -r requirements.txt
python fetch_data.py      # pulls only new games
python fair_odds.py       # output/fair_odds_round_N.csv
python fair_odds.py --odds my_odds.csv --bank 500   # check bookie prices (see odds_example.csv)
python build_site.py      # site/index.html, which you can open in a browser
python backtest.py        # re-tune the model, about once a season (about 5 min)
```

## How the model works

Expected goals = team's expected shots × player's share of team shots × player's goal accuracy.

- **Team shots:** recent shots for, the opponent's recent shots conceded, and home ground.
- **Share:** the player's recent share of team shots, pulled toward the average for the position she's named in. The 21 named players' shares add up to 100%, so a missing forward's shots go to whoever is named.
- **Accuracy:** goals ÷ shots, pulled toward the league rate (about 41%).
- **Goals:** modelled with a negative binomial distribution, which gives P(1+), P(2+) and P(3+).

Every backtest prediction uses only games played before it. Tuned on 2023–25; 2026 is out of sample. In 2026 the model's log loss is 0.416, against 0.453 for position averages and 0.589 for raw goals per game.

## Limits

- **No edge against bookies has been shown yet.** There's no historical goalkicker odds data to test against. Record the odds you take and the closing odds, and look for a consistent pattern of beating the closing price.
- Weather is the AFL feed's forecast label, and its effect is small. In properly wet conditions, shade your expectations down yourself.
- The model only learns about late outs, tags or role changes once they show up on the team sheet or in the stats.
