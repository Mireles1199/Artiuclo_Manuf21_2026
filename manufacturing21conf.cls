% manufacturing21conf.cls -- Class for Manufacturing’21 Toulon template
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{manufacturing21conf}[2025/11/27 Manufacturing’21 Toulon Conference Class]

% Base class: single-column article on A4 paper, 12pt
\LoadClass[12pt,a4paper]{article}

% ---------------------------
%       REQUIRED PACKAGES
% ---------------------------
\RequirePackage[a4paper,
                top=25mm,
                bottom=25mm,
                left=25mm,
                right=25mm]{geometry}

\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{newtxtext,newtxmath}     % Times-like text + math (pdfLaTeX-safe)
\RequirePackage{fancyhdr}
\RequirePackage{caption}
\RequirePackage{xcolor}
\RequirePackage{subcaption}
\RequirePackage{import}

\RequirePackage{manufacturing21} % our .sty file

% \RequirePackage[hidelinks]{hyperref}
\RequirePackage[colorlinks=true,allcolors=blue]{hyperref}
\RequirePackage{enumitem}




% Remove paragraph indentation globally
\setlength{\parindent}{0pt}






% ---------------------------
%       LINE SPACING
% ---------------------------
\AtBeginDocument{%
  \setstretch{1.15}% global line spacing 1.15


% ----------------------------------------
% Display math spacing (equations)
% ----------------------------------------
  \setlength{\abovedisplayskip}{4pt}        % espacio arriba
  \setlength{\belowdisplayskip}{4pt}        % espacio abajo
  \setlength{\abovedisplayshortskip}{4pt}   % arriba si la línea previa es corta
  \setlength{\belowdisplayshortskip}{4pt}   % abajo si la línea previa es corta
}

% ---------------------------
%       HEADER 
% ---------------------------
\setlength{\headheight}{14.5pt} % avoid fancyhdr warnings

\pagestyle{fancy}
\fancyhf{} % clear header and footer fields

% Header: Times 10 pt (footnotesize), italic
\fancyhead[L]{\textit{\footnotesize Conférence Manufacturing'21}}
\fancyhead[R]{\textit{\footnotesize Toulon, 1er et 2 avril 2026}}

% Horizontal rule under the header
\renewcommand{\headrulewidth}{0.4pt}

% No footer rule
\renewcommand{\footrulewidth}{0pt}

% ---------------------------
%            FOOTER
% ---------------------------

\fancyfoot{} % clear footer

% Thin black line above page number
\renewcommand{\footrulewidth}{0.4pt} % 0.4pt black line

% Centered page number
\fancyfoot[C]{\thepage}

% ---------------------------
%        SECTIONING
% ---------------------------
\makeatletter

% Section: bold, normal size, spacing above/below
\renewcommand\section{%
  \@startsection{section}{1}{0pt}%
    {12pt plus 4pt minus 2pt}%   % space before
    {6pt plus 2pt minus 2pt}%    % space after
    {\normalfont\normalsize\bfseries}%  % font
}

% Subsection: italic, same size as body text
\renewcommand\subsection{%
  \@startsection{subsection}{2}{0pt}%
    {10pt plus 4pt minus 2pt}% space above
    {4pt plus 2pt minus 2pt}% space below
    {\normalfont\normalsize\itshape}% italic instead of bold
}

% Subsubsection: italic, same size as body text
\renewcommand\subsubsection{%
  \@startsection{subsubsection}{3}{0pt}%
    {8pt plus 4pt minus 2pt}% space above
    {3pt plus 2pt minus 2pt}% space below
    {\normalfont\normalsize}
}

% ---------------------------------------------------------
% AFFILIATIONS 
% ---------------------------------------------------------

% Store affiliations text (similar to \author)
\newcommand{\@affiliations}{}%
\newcommand{\affiliations}[1]{\def\@affiliations{#1}}%
\newcommand{\affilLine}[1]{#1\\}

% Store email line
\newcommand{\@mail}{}%
\newcommand{\email}[1]{\def\@mail{#1}}%

% ---------------------------
%         TITLE BLOCK
% ---------------------------
\renewcommand{\maketitle}{%
  % Adjust vertical spacing between header and banner
  \vspace*{-30pt}%
  % Conference banner (full width)
  \noindent\includegraphics[width=\textwidth]{Images/Manufacturin21_log.png}\par
  \vspace{-4pt}% space below banner

% Title and authors
\begin{center}
    {\fontsize{22pt}{24pt}\selectfont\bfseries \@title\par}
    \vspace{22pt}% space below the title
    {\fontsize{12pt}{14pt}\selectfont\bfseries \@author\par}
    \vspace{1pt}% space between authors and affiliations
    {%
      \fontsize{10pt}{12pt}\selectfont\itshape
      \setlength{\baselineskip}{20pt}% control line spacing here
      \@affiliations\par
    }
    \vspace{6pt}% space between affiliations and mail
    {%
        \fontsize{10pt}{12pt}\selectfont\itshape
        \@mail\par
    }
\end{center}

  \vspace{16pt}%
}

% ---------------------------
%         CAPTIONS
% ---------------------------
\DeclareCaptionLabelFormat{manufacturing}{#1~#2}

\captionsetup{
  labelformat=manufacturing,
  labelsep=space
}

% Figures: centered, italic caption, 12pt skip below
\captionsetup[figure]{%
  font=it,
  justification=centering,
  singlelinecheck=false,
  skip=5pt
}

% Tables: caption above, left aligned, 12pt skip
\captionsetup[table]{%
  justification=raggedright,
  singlelinecheck=false,
  skip=12pt
}


%       REFERENCES
% --------------------------------
\newcommand{\figref}[1]{(Figure~\ref{#1})}
\newcommand{\tabref}[1]{(Table~\ref{#1})}
\newcommand{\ecref}[1]{(Eq.~\ref{#1})}

% ---------------------------
%        BIBLIOGRAPHY
% ---------------------------
\RequirePackage[
backend=biber,
  style=authoryear,
  sorting=nyt,
  maxcitenames=1,
  mincitenames=1,
  maxbibnames=99,
  uniquelist=false,
  uniquename=false
]{biblatex}

\addbibresource{references.bib} % Add your .bib file here

\renewbibmacro*{name:andothers}{}

\newcommand{\manufacturing@bibliofont}{%
  \fontsize{10}{11.5}\selectfont\itshape
}

\renewenvironment{thebibliography}[1]{%
  \section*{References}%
  \addcontentsline{toc}{section}{References}%
  \begingroup
  \manufacturing@bibliofont
  \list{\@biblabel{\@arabic\c@enumiv}}{%
    \settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \setlength\itemsep{0pt}%
    \setlength\parsep{0pt}%
  }%
}{%
  \endlist
  \endgroup
}


% Bibliography style to match Manufacturing'21 template
\renewcommand*{\bibfont}{%
  \fontsize{10pt}{11.5pt}\selectfont\itshape
}

\setlength{\bibitemsep}{0pt}   % no extra space between items
\setlength{\bibhang}{0pt} 



\newif\ifshownotes
\shownotestrue      % mostrar notas durante redacción
% \shownotesfalse   % ocultar notas en la versión final

% ===== Show / hide notes =====
\newif\ifshownotes

% ---------------------------
%       TODO NOTES SETUP  
% ---------------------------
% ===== todonotes package =====
\RequirePackage[colorinlistoftodos]{todonotes}

% ===== Custom commands (all inline) =====
\newcommand{\cor}[1]{%
  \ifshownotes
    \todo[inline, color=red!30]{Correction: #1}%
  \fi
}

\newcommand{\modi}[1]{%
  \ifshownotes
    \todo[inline, color=blue!30]{Modification: #1}%
  \fi
}

\newcommand{\acl}[1]{%
  \ifshownotes
    \todo[inline, color=green!30]{Clarification: #1}%
  \fi
}


\makeatother

\endinput
