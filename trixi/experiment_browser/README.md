![](http://i.imgur.com/DUiL9yn.png)

## Production mode

To get started, first install all the necessary dependencies.
```
> npm install gulp-cli -g
> npm install
```

Verify Gulp has installed with this:
```
> gulp -v
```

To be able to start the production mode, first run an initial webpack build
```
> npx webpack
```

Afterwards run
```
> python serve_frontend.py
```

## Develop

To get started, first install all the necessary dependencies.
```
> npm install
```

Start the development server (changes will now update live in browser)
```
> npm start
```
To view your project for developing purpose, go to: [http://localhost:3000/](http://localhost:3000/)


