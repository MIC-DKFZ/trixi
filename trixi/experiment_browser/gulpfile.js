
 // modules
var gulp = require('gulp')
var newer = require('gulp-newer')
var imagemin = require('gulp-imagemin')
var htmlclean = require('gulp-htmlclean')
var concat = require('gulp-concat')
var deporder = require('gulp-deporder')
var stripdebug = require('gulp-strip-debug')
var uglify = require('gulp-uglify')
var sass = require('gulp-sass')
var postcss = require('gulp-postcss')
var assets = require('postcss-assets')
var autoprefixer = require('autoprefixer')
var mqpacker = require('css-mqpacker')
var cssnano = require('cssnano')
var inject = require('gulp-inject');
var mainBowerFiles = require('main-bower-files');
var es = require('event-stream')
var fs = require('fs')
var exec = require('child_process').exec;
var runSequence = require('run-sequence');

// development mode?
var devBuild = (process.env.NODE_ENV !== 'production')

// folders
var folder = {
    src: 'src/',
    build: 'build/'
}

gulp.task('create-folder-structure', function() {
	var folders = ["css", "html", "images", "js"]
	if(!fs.existsSync(folder.build)) {
			fs.mkdirSync(folder.build)
		}

	folders.forEach(function(sub_folder) {
		var new_dir = folder.build + sub_folder
		if(!fs.existsSync(new_dir)) {
			fs.mkdirSync(new_dir)
		}
	})
});

gulp.task('bower-install', function (cb) {
  exec('bower install', function (err, stdout, stderr) {
    console.log(stdout);
    console.log(stderr);
    cb(err);
  });
})

// image processing
gulp.task('images', function() {
  var out = folder.build + 'images/';
  return gulp.src(folder.src + 'images/**/*')
    .pipe(newer(out))
    .pipe(imagemin({ optimizationLevel: 5 }))
    .pipe(gulp.dest(out));
});

// HTML processing
gulp.task('html', ['images'], function() {
  var
    out = folder.build + 'html/',
    page = gulp.src(folder.src + 'html/**/*')
      .pipe(newer(out));

  // minify production code
  if (!devBuild) {
    page = page.pipe(htmlclean());
  }

  return page.pipe(gulp.dest(out));
});

// JavaScript processing
gulp.task('js', function() {

  var jsbuild = gulp.src(folder.src + 'js/**/*')
    .pipe(deporder())
    .pipe(concat('main.js'));

  if (!devBuild) {
    jsbuild = jsbuild
      .pipe(stripdebug())
      .pipe(uglify());
  }

  return jsbuild.pipe(gulp.dest(folder.build + 'js/'));

});

// CSS processing
gulp.task('css', ['images'], function() {

  var postCssOpts = [
  assets({ loadPaths: ['images/'] }),
  autoprefixer({ browsers: ['last 2 versions', '> 2%'] }),
  mqpacker
  ];

  if (!devBuild) {
    postCssOpts.push(cssnano);
  }

  return gulp.src(folder.src + 'css/main.css')
    .pipe(sass({
      outputStyle: 'nested',
      imagePath: 'images/',
      precision: 3,
      errLogToConsole: true
    }))
    .pipe(postcss(postCssOpts))
    .pipe(gulp.dest(folder.build + 'css/'));

});

gulp.task('inject-overview', function() {
	var target = gulp.src(folder.src + 'html/overview.html')
	var bowerStream = gulp.src(mainBowerFiles(), {read: false})
					//.pipe(gulp.dest(folder.build + 'html/'))
	var jsStream= gulp.src(folder.src + 'js/**/*')
				.pipe(deporder())
				.pipe(concat('concatenated_javascript.js'))
				.pipe(gulp.dest(folder.build + 'js/'))

	var cssStream= gulp.src(folder.src + 'css/**/*')
				.pipe(deporder())
				.pipe(concat('concatenated_css.css'))
				.pipe(gulp.dest(folder.build + 'css/'))

    //var output = target.pipe(inject(es.merge(bowerStream, jsStream))).pipe(gulp.dest(folder.build + 'html/'), {name: 'bower'});
    var output = target.pipe(inject(es.merge(bowerStream, jsStream, cssStream), {addRootSlash : false, relative:true}))
    				   .pipe(gulp.dest(folder.build + 'html/'), {name: 'bower'});

	return output
});

gulp.task('inject-experiment', function() {
	var target = gulp.src(folder.src + 'html/experiment.html')
	
	//var sources = gulp.src(mainBowerFiles(), {read: false})
	//var output = target.pipe(inject(sources),{relative: true}).pipe(gulp.dest(folder.build + 'html/'), {name: 'bower'});
	var bowerStream = gulp.src(mainBowerFiles(), {read: false})
					//.pipe(gulp.dest(folder.build + 'html/'))
	var jsStream= gulp.src(folder.src + 'js/**/*')
				.pipe(deporder())
				.pipe(concat('concatenated_javascript.js'))
				.pipe(gulp.dest(folder.build + 'js/'))

	var cssStream= gulp.src(folder.src + 'css/**/*')
				.pipe(deporder())
				.pipe(concat('concatenated_css.css'))
				.pipe(gulp.dest(folder.build + 'css/'))

    //var output = target.pipe(inject(es.merge(bowerStream, jsStream))).pipe(gulp.dest(folder.build + 'html/'), {name: 'bower'});
    var output = target.pipe(inject(es.merge(bowerStream, jsStream, cssStream), {addRootSlash : false, relative:true}))
    				   .pipe(gulp.dest(folder.build + 'html/'), {name: 'bower'});

	return output
});

gulp.task('inject', ['inject-overview', 'inject-experiment'])

gulp.task('prepeare-structure', function(done) {
    runSequence('create-folder-structure', 'bower-install', function() {
        console.log('Preparation done!');
        done();
    });
});

// run all tasks
gulp.task('run', ['html', 'css', 'inject']);

// watch for changes
gulp.task('watch', function() {

  // image changes
  gulp.watch(folder.src + 'images/**/*', ['images']);

  // html changes
  gulp.watch(folder.src + 'html/**/*', ['html']);

  // javascript changes
  gulp.watch(folder.src + 'js/**/*', ['js']);

  // css changes
  gulp.watch(folder.src + 'scss/**/*', ['css']);

});

// default task
gulp.task('default', ['prepeare-structure', 'run', 'watch']);