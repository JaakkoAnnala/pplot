_completion_pplot()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"

    # pp -help | grep -Po '^\s+\-\S+'
    opts=" -h -c -x -e -p -s -head -cr -logy -logx -lny -lnx -xlim -ylim -hist -norm -fold -we -subf -subc -expr -b -be -int -rf -surf -cmap -logc -sc -title -xlab -ylab -ps -ts -te -axvl -axhl -mean -err -skip_lines -list_expr_funcs"

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}

complete -F _completion_pplot pp