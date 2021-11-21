`gspread` authentication tokens expire after a week.  You need to manually 
remove the authentication file to repeat the authentication process:

```
rm ~/.config/gspread/authorized_user.json
```
