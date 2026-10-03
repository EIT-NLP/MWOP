import argparse
import sys

def main():
    argv = sys.argv[1:]
    if argv and (argv[0].startswith('-') and argv[0] not in ('--help', '-h') or argv[0] == 'eval'):
        sys.argv = [sys.argv[0]] + (argv[1:] if argv[0] == 'eval' else argv)
        from lmms_eval.__main__ import cli_evaluate
        cli_evaluate()
        return
    parser = argparse.ArgumentParser(prog='lmms-eval', description='OV evaluation: eval, tasks, models, version')
    sub = parser.add_subparsers(dest='subcommand')
    from lmms_eval.cli.tasks_cmd import add_tasks_parser
    from lmms_eval.cli.models_cmd import add_models_parser
    from lmms_eval.cli.version_cmd import add_version_parser
    add_tasks_parser(sub)
    add_models_parser(sub)
    add_version_parser(sub)
    args = parser.parse_args(argv)
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
