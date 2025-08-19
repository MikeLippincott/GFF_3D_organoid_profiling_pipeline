To deploy the shiny app, ensure that the correct environemnt is installed by running the following command:
```bash
mamba env create -f <github_repo>/environments/R_shiny.yml
```

Then, run the app using:
```bash
source 6.dynamic_viz_Rshiny/deploy_app.sh
```

You will also need to set up the `.Renviron` file with the necessary environment variables. This file should contain:
```
RSCONNECT_NAME="your_account_name
RSCONNECT_TOKEN="your_token"
RSCONNECT_SECRET="your_secret"
```

This file is used to authenticate your deployment to the shiny server. Make sure to replace `your_account_name`, `your_token`, and `your_secret` with your actual RStudio Connect account details.
This file should not be committed to version control as it contains sensitive information.
Make sure to add `.Renviron` to your `.gitignore` file to prevent it from being tracked by git.

Note: The app is optionally hosted for free on shinyapps.io, which is a service provided by Posit.
The free tier has some limitations, such as a maximum of 25 active hours per month and a maximum of 5 applications.
You can upgrade to a paid plan if you need more resources or features.
