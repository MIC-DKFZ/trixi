const HtmlWebpackPlugin = require('html-webpack-plugin'); // Require  html-webpack-plugin plugin

module.exports = {
  mode: 'development',
  //mode: 'production',
  entry: __dirname + "/src/app/overview.js", // webpack entry point. Module to start building dependency graph
  output: {
    path: __dirname + '/dist', // Folder to store generated bundle
    filename: 'bundle.js',  // Name of generated bundle after build
    publicPath: '/' // public URL of the output directory when referenced in a browser
  },
  module: {  // where we defined file patterns and their loaders

      rules: [
          {
            test: /\.js$/,
            use: 'babel-loader',
            exclude: [
              /node_modules/
            ]
          },
          {
              test: /\.scss$/,
              use: ['style-loader', 'css-loader']
          },
          {
              test: /\.css$/,
              use: ['style-loader', 'css-loader']
          },
          {
              test: /\.svg$/,
              loader: 'svg-inline-loader'
          },
          {
              test: /\.(woff(2)?|ttf|eot|svg)(\?v=\d+\.\d+\.\d+)?$/,
              use: [{
                  loader: 'file-loader',
                  options: {
                      name: '[name].[ext]',
                      outputPath: 'fonts/'
                  }
                }]
          }
          /*
          {
              test: /\.ttf$/,
              use: [
                  {
                      loader: 'ttf-loader',
                      options: {
                          name: './font/[hash].[ext]',
                      },
                  },
              ]
          },
           */
      ]
  },
  plugins: [  // Array of plugins to apply to build chunk
      new HtmlWebpackPlugin({
          template: __dirname + "/src/public/overview.html",
          inject: 'body'
      })
  ],
  devServer: {  // configuration for webpack-dev-server
      contentBase: './src/public',  //source of static assets
      port: 7700, // port to run dev-server
  } 
};