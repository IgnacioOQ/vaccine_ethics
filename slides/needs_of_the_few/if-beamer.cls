% ------------------------------------------------------------------------------- %
%                                    LICENSE   	                                  %
% ------------------------------------------------------------------------------- %
%                     GNU GENERAL PUBLIC LICENSE                                  %
%                        Version 3, 29 June 2007                                  %
% Author: Humberto da Silva Neto                                                  %
% Git Repo: https://github.com/hsneto/if-beamer/                                  %
% ------------------------------------------------------------------------------- %

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{if-beamer}[10/09/2018, v1.0.1]

\PassOptionsToPackage{svgnames}{xcolor}
\LoadClass[compress, 8pt]{beamer}

% --------------------------------------------------- %
%                       Packages  	                  %
% --------------------------------------------------- %
\usepackage[brazilian, english]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\usepackage{pdfpages}
\usepackage{graphicx} % http://ctan.org/pkg/graphicx
\usepackage[footnotesize,hang]{caption}
\usepackage{hyperref}
\usepackage{tikz}
\usepackage{listings}
\usepackage{etoolbox}
\usepackage{tcolorbox}
\usepackage{tabularx}
\usepackage{array}
\usepackage{colortbl}
\tcbuselibrary{skins}
\usepackage{ragged2e}
\usepackage{courier}

% --------------------------------------------------- %
%                   Basic colors	                  %
% --------------------------------------------------- %
% Blue
\definecolor{airforceblue}{rgb}{0.36, 0.54, 0.66}
\definecolor{aliceblue}{rgb}{0.94, 0.97, 1.0}
% Green
\definecolor{cadmiumgreen}{rgb}{0.0, 0.42, 0.24}
\definecolor{honeydew}{rgb}{0.94, 1.0, 0.94}
% Red
\definecolor{carnelian}{rgb}{0.7, 0.11, 0.11}
\definecolor{melon}{rgb}{0.99, 0.74, 0.71}
% Orange
\definecolor{orange-red}{rgb}{1.0, 0.27, 0.0}
\definecolor{peach-orange}{rgb}{1.0, 0.8, 0.6}
% Yellow
\definecolor{pastelyellow}{rgb}{1.0, 0.92, 0.71}
\definecolor{uscgold}{rgb}{1.0, 0.8, 0.0}
% Purple
\definecolor{purpleheart}{rgb}{0.41, 0.21, 0.61}
\definecolor{lightmauve}{rgb}{0.86, 0.82, 1.0}
% Brown
\definecolor{darkbrown}{rgb}{0.4, 0.26, 0.13}
\definecolor{moccasin}{rgb}{0.98, 0.92, 0.84}
% Grey scale
\definecolor{ashgrey}{rgb}{0.38, 0.38, 0.38}
\definecolor{lightgray}{rgb}{0.90, 0.90, 0.90}
\definecolor{cadetgrey}{rgb}{0.57, 0.64, 0.69}
% Black and white
\definecolor{black}{rgb}{0.0, 0.0, 0.0}
\definecolor{white}{rgb}{1.0, 1.0, 1.0}

% --------------------------------------------------- %
%                     Set colors	                  %
% --------------------------------------------------- %
\colorlet{mainColor1}{cadmiumgreen} % dark
\colorlet{mainColor2}{honeydew} % light

\colorlet{bodyColor}{white}
\colorlet{titleColor}{mainColor1}
\colorlet{chipColor}{mainColor1}

\colorlet{sectionColor}{black}
\colorlet{subsectionColor}{ashgrey}

\colorlet{authorFootlineColor}{black}
\colorlet{instituteFootlineColor}{black}
\colorlet{titleFootlineColor}{ashgrey}
\colorlet{pageFootlineColor}{black}
\colorlet{dateFootlineColor}{black}

\colorlet{normalTitleBlockColor}{mainColor1}
\colorlet{normalBlockColor}{mainColor2}
\colorlet{exampleTitleBlockColor}{airforceblue}
\colorlet{exampleBlockColor}{aliceblue}
\colorlet{alertTitleBlockColor}{carnelian}
\colorlet{alertBlockColor}{melon}

\definecolor{CodeGreen}{rgb}{0,0.6,0}
\definecolor{CodePurple}{rgb}{0.58,0,0.82}
\definecolor{CodeOrange}{rgb}{1,.45,0}
\definecolor{CodeGrey}{rgb}{0.92,0.92,0.92}

\colorlet{normalTextColor}{black}
\colorlet{titleTextColor}{white}
\colorlet{chipTextColor}{white}
\colorlet{sectionTextColor}{white}
\colorlet{subsectionTextColor}{white}
\colorlet{exampleTextColor}{airforceblue}
\colorlet{emphaseTextColor}{orange-red}
\colorlet{alertTextColor}{carnelian}
\colorlet{authorFootlineTextColor}{white}
\colorlet{instituteFootlineTextColor}{white}
\colorlet{titleFootlineTextColor}{white}
\colorlet{pageFootlineTextColor}{white}
\colorlet{dateFootlineTextColor}{white}
\colorlet{blockTitleTextColor}{white}
\colorlet{blockBodyTextColor}{black}
\colorlet{tableTitleTextColor}{white}
\colorlet{tableBodyTextColor}{black}

% --------------------------------------------------- %
%                      Structure                      %
% --------------------------------------------------- %
\usecolortheme[named=mainColor1]{structure}
\useinnertheme{rectangles}
\useoutertheme[subsection=false]{miniframes}
\setbeamertemplate{navigation symbols}{}

\setbeamercolor{titlelike}{fg=titleTextColor,bg=titleColor}
\setbeamercolor*{normal text}{fg=normalTextColor,bg=bodyColor}
\setbeamercolor*{example text}{fg=exampleTextColor}
\setbeamercolor*{alerted text}{fg=alertTextColor}

\newcommand{\exemple}[1]{{\color{example text.fg} #1}}
\renewcommand{\emph}[1]{\textcolor{orange-red}{\textbf{#1}}}

\setbeamercolor{itemize item}{fg=titleColor!70}
\setbeamercolor{enumerate item}{fg=titleColor!70}
\setbeamercolor{description item}{fg=titleColor!70}

\setbeamercolor{section in head/foot}{bg=sectionColor, fg=sectionTextColor}
\setbeamercolor{subsection in head/foot}{bg=subsectionColor, fg=subsectionTextColor}

\useinnertheme{default}
\setbeamercolor{item projected}{bg=chipColor,fg=chipTextColor}
\beamer@compresstrue
\defbeamertemplate*{headline}{smoothbars theme}{%
  \begin{beamercolorbox}[ht=2.125ex,dp=3.150ex]{section in head/foot}
  \insertnavigation{\paperwidth}
  \end{beamercolorbox}%

  \begin{beamercolorbox}[ht=2.125ex,dp=1.125ex,%
  leftskip=.3cm,rightskip=.3cm plus1fil]{subsection in head/foot}
  \usebeamerfont{subsection in head/foot}\insertsubsectionhead
  \end{beamercolorbox}%
}

\setbeamercolor{author in head/foot}{bg=authorFootlineColor, fg=authorFootlineTextColor}
\setbeamercolor{title in head/foot}{bg=titleFootlineColor, fg=titleFootlineTextColor}
\setbeamercolor{institute in head/foot}{bg=instituteFootlineColor, fg=instituteFootlineTextColor}
\setbeamercolor{date in head/foot}{bg=dateFootlineColor, fg=dateFootlineTextColor}

\defbeamertemplate*{footline}{infolines theme}{
\leavevmode%
\hbox{%
\begin{beamercolorbox}[wd=.20\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
\usebeamerfont{author in head/foot}\insertshortauthor~~
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.10\paperwidth,ht=2.25ex,dp=1ex,center]{institute in head/foot}%
\usebeamerfont{institute in head/foot}\insertshortinstitute
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.40\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
\usebeamerfont{title in head/foot}\insertshorttitle
\end{beamercolorbox}%

\begin{beamercolorbox}[wd=.30\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
\usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{2em}
\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
\end{beamercolorbox}
}%
\vskip0pt%
}

\mode
<all>

% --------------------------------------------------- %
%                  Renew Commands                     %
% --------------------------------------------------- %
\apptocmd{\frame}{}{\justifying}{}
\renewcommand{\arraystretch}{1.5}
\newcommand{\urlcolor}[1]{\textcolor{magenta}{#1}}
\newcommand{\cref}[2]{\urlcolor{\href{#1}{#2}}}
\newcommand{\code}[1]{\codebox{\centering\texttt{#1}}}
\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}

% --------------------------------------------------- %
%                  Sections Frames                    %
% --------------------------------------------------- %
\AtBeginSection[]{
  \begin{frame}
  \vfill
  \centering
  \begin{beamercolorbox}[sep=8pt,center,shadow=true,rounded=true]{title}
    \usebeamerfont{title}\insertsectionhead\par%
  \end{beamercolorbox}
  \vfill
  \end{frame}
}

% --------------------------------------------------- %
%                        Tables                       %
% --------------------------------------------------- %
\newcolumntype{Y}{>{\raggedleft\arraybackslash}X}

% --------------------------------------------------- %
%                        Blocks                       %
% --------------------------------------------------- %
\setbeamertemplate{blocks}[rectangle]

\setbeamercolor*{block title}{fg=blockTitleTextColor, bg=normalTitleBlockColor}
\setbeamercolor*{block body}{fg=blockBodyTextColor, bg=normalBlockColor}
\setbeamercolor*{block title alerted}{fg=blockTitleTextColor, bg=alertTitleBlockColor}
\setbeamercolor*{block body alerted}{fg=blockBodyTextColor, bg=alertBlockColor}
\setbeamercolor*{block title example}{fg=blockTitleTextColor, bg=exampleTitleBlockColor}
\setbeamercolor*{block body example}{fg=blockBodyTextColor, bg=exampleBlockColor}
\setbeamerfont{block title}{size={}}

\AtBeginEnvironment{block}{%
  \setbeamercolor{itemize item}{fg=normalTitleBlockColor!70}
}
\AtBeginEnvironment{exampleblock}{%
  \setbeamercolor{itemize item}{fg=exampleTitleBlockColor!70}
}
\AtBeginEnvironment{alertblock}{%
  \setbeamercolor{itemize item}{fg=alertTitleBlockColor!70}
}

% --------------------------------------------------- %
%                      Listings 	                  %
% --------------------------------------------------- %
\definecolor{backcolour}{RGB}{245,245,245}
\definecolor{commentcolour}{RGB}{0,128,0}
\definecolor{keywordcolour}{RGB}{249,38,114}
\definecolor{stringcolour}{RGB}{255,102,0}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},
    commentstyle=\color{commentcolour},
    keywordstyle=\color{keywordcolour}\bfseries,
    numberstyle=\tiny\color{black},
    stringstyle=\color{stringcolour},
    emphstyle=\color{red},
    basicstyle=\footnotesize\ttfamily,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=t,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}
\lstset{style=mystyle}
