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

Run the following command to download all dependecies and build the frontend
```
> gulp install
```

## Develop

Perform the same steps like for production mode. 

Run just gulp to build the frontend and start to watch for file changes
```
> gulp
```

Open any build html file in `build/html` in your browser. Any change in a file will automatically cause a rebuild.

Please make sure you just edit files in the `src/` directory. Any changes done in the `build/` directory will be overwritten.


